import pytest

from studybuddy.config import DEFAULT_BASE_URL, load_settings, validate_dirs
from studybuddy.errors import ConfigError


def make_env(tmp_path, **over):
    env = {
        "SB_READER_MODEL": "reader",
        "SB_CHAT_MODEL": "chat",
        "SB_EMBED_MODEL": "embed",
        "SB_LIBRARY_DIR": str(tmp_path / "lib"),
        "SB_DATA_DIR": str(tmp_path / "data"),
    }
    env.update(over)
    return env


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    r.mkdir()
    return r


def load(tmp_path, repo, **over):
    return load_settings(make_env(tmp_path, **over), repo_root=repo)


def test_defaults(tmp_path, repo):
    s = load(tmp_path, repo)
    assert s.base_url == DEFAULT_BASE_URL
    assert s.tag_timeout_seconds == 300
    assert s.telegram_user_id is None
    assert s.subjects_file == repo / "subjects.json"


def test_overrides_and_trailing_slash(tmp_path, repo):
    s = load(tmp_path, repo, SB_LMSTUDIO_BASE_URL="http://h:1/v1/", TELEGRAM_USER_ID="42",
             SB_TAG_TIMEOUT_SECONDS="60")
    assert s.base_url == "http://h:1/v1"
    assert s.telegram_user_id == 42
    assert s.tag_timeout_seconds == 60


@pytest.mark.parametrize("name", ["SB_READER_MODEL", "SB_CHAT_MODEL", "SB_EMBED_MODEL",
                                  "SB_LIBRARY_DIR", "SB_DATA_DIR"])
def test_missing_required(tmp_path, repo, name):
    env = make_env(tmp_path)
    del env[name]
    with pytest.raises(ConfigError, match=name):
        load_settings(env, repo_root=repo)


@pytest.mark.parametrize("name", ["SB_TAG_TIMEOUT_SECONDS", "TELEGRAM_USER_ID"])
def test_bad_numbers(tmp_path, repo, name):
    with pytest.raises(ConfigError, match=name):
        load(tmp_path, repo, **{name: "abc"})


def test_validate_creates_dirs(tmp_path, repo):
    s = load(tmp_path, repo)
    validate_dirs(s, repo_root=repo)
    assert s.library_dir.is_dir() and s.data_dir.is_dir()


def test_library_inside_repo_rejected(tmp_path, repo):
    s = load(tmp_path, repo, SB_LIBRARY_DIR=str(repo / "lib"))
    with pytest.raises(ConfigError, match="outside the repo"):
        validate_dirs(s, repo_root=repo)


def test_repo_inside_data_rejected(tmp_path, repo):
    s = load(tmp_path, repo, SB_DATA_DIR=str(tmp_path))
    with pytest.raises(ConfigError, match="outside the repo"):
        validate_dirs(s, repo_root=repo)


def test_nested_dirs_rejected(tmp_path, repo):
    s = load(tmp_path, repo, SB_DATA_DIR=str(tmp_path / "lib" / "data"))
    with pytest.raises(ConfigError, match="nested"):
        validate_dirs(s, repo_root=repo)


def test_uncreatable_dir(tmp_path, repo):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    s = load(tmp_path, repo, SB_LIBRARY_DIR=str(blocker / "lib"))
    with pytest.raises(ConfigError, match="SB_LIBRARY_DIR"):
        validate_dirs(s, repo_root=repo)


def test_embed_prefixes_from_env(tmp_path, repo):
    s = load(tmp_path, repo, EMBED_DOC_PREFIX="search_document: ", EMBED_QUERY_PREFIX="search_query: ")
    assert (s.embed_doc_prefix, s.embed_query_prefix) == ("search_document: ", "search_query: ")


def test_embed_prefixes_default_empty(tmp_path, repo):
    s = load(tmp_path, repo)
    assert (s.embed_doc_prefix, s.embed_query_prefix) == ("", "")


def test_quoted_prefixes_keep_trailing_space_from_dotenv_file(tmp_path, repo, monkeypatch):
    for name in ("SB_READER_MODEL", "SB_CHAT_MODEL", "SB_EMBED_MODEL", "SB_LIBRARY_DIR",
                 "SB_DATA_DIR", "EMBED_DOC_PREFIX", "EMBED_QUERY_PREFIX"):
        monkeypatch.delenv(name, raising=False)  # restored on teardown; load_dotenv mutates os.environ
    (repo / ".env").write_text(
        "SB_READER_MODEL=r\nSB_CHAT_MODEL=c\nSB_EMBED_MODEL=e\n"
        f"SB_LIBRARY_DIR={(tmp_path / 'lib').as_posix()}\nSB_DATA_DIR={(tmp_path / 'data').as_posix()}\n"
        'EMBED_DOC_PREFIX="search_document: "\nEMBED_QUERY_PREFIX="search_query: "\n',
        encoding="utf-8",
    )
    s = load_settings(repo_root=repo)
    assert s.embed_doc_prefix == "search_document: "
    assert s.embed_query_prefix == "search_query: "


def test_reader_settings_defaults(tmp_path, repo):
    s = load(tmp_path, repo)
    assert s.reader_max_edge == 2048
    assert s.samples_dir is None


def test_reader_settings_override(tmp_path, repo):
    s = load(tmp_path, repo, SB_READER_MAX_EDGE="1600", SB_SAMPLES_DIR=str(tmp_path / "samples"))
    assert s.reader_max_edge == 1600
    assert s.samples_dir == (tmp_path / "samples").resolve()


@pytest.mark.parametrize("value", ["abc", "100"])
def test_reader_max_edge_invalid(tmp_path, repo, value):
    with pytest.raises(ConfigError, match="SB_READER_MAX_EDGE"):
        load(tmp_path, repo, SB_READER_MAX_EDGE=value)


def test_samples_dir_inside_repo_rejected(tmp_path, repo):
    s = load(tmp_path, repo, SB_SAMPLES_DIR=str(repo / "samples"))
    with pytest.raises(ConfigError, match="SB_SAMPLES_DIR"):
        validate_dirs(s, repo_root=repo)


def test_missing_samples_dir_is_not_created(tmp_path, repo):
    s = load(tmp_path, repo, SB_SAMPLES_DIR=str(tmp_path / "samples"))
    validate_dirs(s, repo_root=repo)
    assert not (tmp_path / "samples").exists()
