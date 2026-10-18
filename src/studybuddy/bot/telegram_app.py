"""The Telegram side: python-telegram-bot, long polling (no webhook, no open port), one allowed user.

`TelegramBot` runs in its own thread with its own event loop so it can live inside the web app's process. The handlers here only
translate Telegram updates into `Incoming` / callback / text calls on `BotFlow`, which holds the conversation.

Who may talk to the bot: ONLY `SB_TELEGRAM_USER_ID`, in a private chat. Everybody else gets no reply at all; the log records
the kind of update and the sender's numeric id, never what was written or sent.
"""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Sequence

from telegram import (InlineKeyboardButton, InlineKeyboardMarkup, ReplyParameters, Update)
from telegram.error import BadRequest, InvalidToken, TelegramError
from telegram.ext import (Application, ApplicationBuilder, ApplicationHandlerStop, CallbackQueryHandler, CommandHandler, ContextTypes,
                          MessageHandler, TypeHandler, filters)
from telegram.request import BaseRequest

from ..config import Settings
from ..ingest.results import IngestResult
from ..ingest.subjects import SubjectList
from .flow import BotFlow, Button, DownloadError, Incoming, TooBigError
from .spool import Spool
from .state import StateStore

log = logging.getLogger(__name__)

TICK_SECONDS = 2.0
CONNECT_RETRY_SECONDS = 60.0
MAX_TEXT = 4000                      # Telegram refuses messages over 4096 characters
IMAGE_MIMES = {"image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff", "image/heic", "image/heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


class PtbMessenger:
    """`Messenger` for the flow, on top of a telegram.Bot."""

    def __init__(self, bot):
        self.bot = bot

    @staticmethod
    def _markup(buttons: Sequence[Sequence[Button]] | None):
        if not buttons:
            return None
        return InlineKeyboardMarkup([[InlineKeyboardButton(b.label, callback_data=b.data) for b in row] for row in buttons])

    async def send(self, chat_id: int, text: str, buttons=None, reply_to: int | None = None) -> int:
        extra = {"reply_parameters": ReplyParameters(message_id=reply_to, allow_sending_without_reply=True)} if reply_to else {}
        message = await self.bot.send_message(chat_id=chat_id, text=text[:MAX_TEXT], reply_markup=self._markup(buttons), **extra)
        return message.message_id

    async def edit(self, chat_id: int, message_id: int, text: str, buttons=None) -> None:
        try:
            await self.bot.edit_message_text(chat_id=chat_id, message_id=message_id, text=text[:MAX_TEXT], reply_markup=self._markup(buttons))
        except BadRequest as exc:
            if "not modified" not in str(exc).lower():
                log.warning("could not edit message %s: %s", message_id, exc)

    async def download(self, file_id: str) -> bytes:
        try:
            file = await self.bot.get_file(file_id)
            return bytes(await file.download_as_bytearray())
        except BadRequest as exc:
            if "too big" in str(exc).lower():
                raise TooBigError(str(exc)) from exc
            raise DownloadError(str(exc)) from exc
        except TelegramError as exc:
            raise DownloadError(str(exc)) from exc


def _local_naive(moment: datetime) -> datetime:
    return moment.astimezone().replace(tzinfo=None) if moment.tzinfo else moment


def _what(document) -> str:
    name = (document.file_name or "").lower()
    suffix = Path(name).suffix
    if suffix:
        return f"a {suffix.lstrip('.').upper()} file"
    return f"a file of type {document.mime_type}" if document.mime_type else "that kind of file"


class TelegramBot:
    def __init__(self, settings: Settings, service, subjects: SubjectList, request: BaseRequest | None = None,
                 clock: Callable[[], float] = time.time, tick_seconds: float = TICK_SECONDS,
                 connect_retry_seconds: float = CONNECT_RETRY_SECONDS):
        if not settings.telegram_token or settings.telegram_user_id is None:
            raise ValueError("the bot needs a token and the student's Telegram user id")
        self.settings, self.service, self.subjects = settings, service, subjects
        self.allowed_id = settings.telegram_user_id
        self._request = request
        self.tick_seconds, self.connect_retry_seconds = tick_seconds, connect_retry_seconds
        self.state = StateStore(settings.data_dir / "bot")
        self.spool = Spool(settings.library_dir)
        self.application: Application | None = None
        self.flow: BotFlow | None = None
        self.status = "not started"
        self.ready = threading.Event()
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop: asyncio.Event | None = None
        self._clock = clock
        self._callback_registered = False
        self._accepting = False                      # results are taken from the ingest worker once the flow has reconciled

    # ---- building the application -----------------------------------------------------------------------

    def build_application(self) -> Application:
        builder = ApplicationBuilder().token(self.settings.telegram_token).concurrent_updates(False)
        if self._request is not None:
            builder = builder.request(self._request).get_updates_request(self._request)
        app = builder.build()
        self.flow = BotFlow(self.service, self.subjects, self.spool, self.state, PtbMessenger(app.bot), self._clock)
        app.add_handler(TypeHandler(Update, self._authorise), group=-1)           # runs first; stops everything for strangers
        app.add_handler(CommandHandler(["start", "help"], self._on_command))
        app.add_handler(MessageHandler(filters.PHOTO, self._on_photo))
        app.add_handler(MessageHandler(filters.Document.ALL, self._on_document))
        app.add_handler(MessageHandler(filters.VIDEO | filters.AUDIO | filters.VOICE | filters.VIDEO_NOTE | filters.ANIMATION
                                       | filters.Sticker.ALL | filters.CONTACT | filters.LOCATION, self._on_other_media))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._on_text))
        app.add_handler(CallbackQueryHandler(self._on_callback))
        app.add_error_handler(self._on_error)
        self.application = app
        return app

    # ---- handlers ---------------------------------------------------------------------------------------

    async def _authorise(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        user, chat = update.effective_user, update.effective_chat
        if user is not None and user.id == self.allowed_id and chat is not None and chat.type == "private":
            return
        kind = next((name for name in ("message", "callback_query", "edited_message", "channel_post", "inline_query", "my_chat_member")
                     if getattr(update, name, None) is not None), "update")
        log.warning("ignored a %s from an unauthorised sender (id %s)", kind, user.id if user is not None else "unknown")
        raise ApplicationHandlerStop                                               # no reply, nothing is stored

    async def _on_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.flow.hint(update.effective_chat.id)

    async def _on_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self.flow.on_text(update.effective_chat.id, update.effective_message.text or "")

    async def _on_photo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        m = update.effective_message
        biggest = m.photo[-1]                                                       # Telegram sends several sizes; the last is the largest
        await self.flow.on_file(Incoming(
            chat_id=m.chat_id, message_id=m.message_id, kind="photo", file_id=biggest.file_id, file_unique_id=biggest.file_unique_id,
            sent_at=_local_naive(m.date), file_size=biggest.file_size, media_group_id=m.media_group_id, caption=m.caption))

    async def _on_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        m, doc = update.effective_message, update.effective_message.document
        suffix = Path((doc.file_name or "").lower()).suffix
        mime = (doc.mime_type or "").lower()
        if mime == "application/pdf" or suffix == ".pdf":
            kind = "pdf"
        elif mime in IMAGE_MIMES or (mime.startswith("image/") and suffix in IMAGE_EXTENSIONS) or (not mime and suffix in IMAGE_EXTENSIONS):
            kind = "photo-file"
        else:
            kind = "unsupported"
        await self.flow.on_file(Incoming(
            chat_id=m.chat_id, message_id=m.message_id, kind=kind, file_id=doc.file_id, file_unique_id=doc.file_unique_id,
            sent_at=_local_naive(m.date), file_size=doc.file_size, file_name=doc.file_name, media_group_id=m.media_group_id,
            caption=m.caption, what=_what(doc)))

    async def _on_other_media(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        m = update.effective_message
        what = ("a video" if m.video or m.video_note else "audio" if m.audio else "a voice message" if m.voice else
                "a GIF" if m.animation else "a sticker" if m.sticker else "that")
        await self.flow.on_file(Incoming(chat_id=m.chat_id, message_id=m.message_id, kind="unsupported", file_id="-",
                                         file_unique_id=f"unsupported-{m.message_id}", sent_at=_local_naive(m.date), what=what))

    async def _on_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        query = update.callback_query
        notice = await self.flow.on_callback(query.message.chat_id, query.message.message_id, query.data or "")
        await query.answer(notice)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        log.error("bot handler failed: %s: %s", type(context.error).__name__, context.error)

    # ---- the thread -------------------------------------------------------------------------------------

    def start(self) -> "TelegramBot":
        if self._thread and self._thread.is_alive():
            return self
        if not self._callback_registered:
            self.service.add_callback(self._on_ingest_result)
            self._callback_registered = True
        self._thread = threading.Thread(target=lambda: asyncio.run(self._main()), name="telegram-bot", daemon=True)
        self._thread.start()
        return self

    def stop(self, timeout: float = 15.0) -> None:
        loop, stop = self._loop, self._stop
        if loop is not None and stop is not None:
            try:
                loop.call_soon_threadsafe(stop.set)
            except RuntimeError:             # the loop already ended (the bot stopped itself, or stop() was called twice)
                pass
        if self._thread:
            self._thread.join(timeout)

    def _on_ingest_result(self, result: IngestResult) -> None:
        """Called on the ingest worker's thread: hand the result to the bot's loop."""
        loop = self._loop
        if not self._accepting or loop is None or not loop.is_running() or self.flow is None:
            return                                   # the bot is not up; `BotFlow.start()` reconciles when it is
        future = asyncio.run_coroutine_threadsafe(self.flow.on_result(result), loop)
        future.add_done_callback(self._log_result_error)

    @staticmethod
    def _log_result_error(future) -> None:
        if not future.cancelled() and future.exception() is not None:
            log.error("could not tell the student about a result: %s: %s", type(future.exception()).__name__, future.exception())

    async def _main(self) -> None:
        self._loop, self._stop = asyncio.get_running_loop(), asyncio.Event()
        app = self.build_application()
        try:
            while not self._stop.is_set():
                try:
                    await app.initialize()                                           # getMe: proves the token and the network
                    break
                except InvalidToken:
                    self.status = "stopped: Telegram does not accept the bot token"
                    log.error("the Telegram bot token was rejected; the bot is off (check TELEGRAM_BOT_TOKEN in .env)")
                    return
                except Exception as exc:
                    self.status = "waiting for Telegram (no connection)"
                    log.warning("cannot reach Telegram yet (%s); trying again in %.0f s", type(exc).__name__, self.connect_retry_seconds)
                    try:
                        await asyncio.wait_for(self._stop.wait(), self.connect_retry_seconds)
                    except asyncio.TimeoutError:
                        pass
            if self._stop.is_set():
                return
            await self.flow.start()
            self._accepting = True
            await app.start()
            await app.updater.start_polling(drop_pending_updates=False, allowed_updates=["message", "callback_query"])
            self.status = f"connected as @{app.bot.username}"
            log.info("Telegram bot is listening (long polling)")
            self.ready.set()
            ticker = asyncio.create_task(self._ticker())
            await self._stop.wait()
            ticker.cancel()
        finally:
            await self._shutdown(app)

    async def _ticker(self) -> None:
        while True:
            try:
                await self.flow.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("bot tick failed")
            await asyncio.sleep(self.tick_seconds)

    async def _shutdown(self, app: Application) -> None:
        self._accepting = False
        for step in (app.updater.stop if app.updater and app.updater.running else None, app.stop if app.running else None, app.shutdown):
            if step is None:
                continue
            try:
                await step()
            except Exception as exc:
                log.warning("bot shutdown step failed: %s", type(exc).__name__)
        if not self.status.startswith("stopped:"):
            self.status = "stopped"
