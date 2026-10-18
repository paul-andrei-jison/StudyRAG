<#
.SYNOPSIS
  Adds or removes a Windows Startup shortcut that runs start.bat when you sign in. Runs only when YOU run it.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\install_startup.ps1 -Action add
  powershell -ExecutionPolicy Bypass -File scripts\install_startup.ps1 -Action status
  powershell -ExecutionPolicy Bypass -File scripts\install_startup.ps1 -Action remove

  add     creates "Study Buddy.lnk" in your Startup folder. It runs `start.bat wait` minimized: it waits up to 10 minutes
          for LM Studio and the models, then starts Study Buddy. Nothing else on the computer is changed.
  remove  deletes that one shortcut (and nothing else).
  status  says whether the shortcut exists and where it points.

  -Folder is for testing: use another folder instead of the real Startup folder.
#>
param(
    [Parameter(Mandatory = $true)][ValidateSet('add', 'remove', 'status')][string]$Action,
    [string]$Folder = [Environment]::GetFolderPath('Startup')
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$target = Join-Path $root 'start.bat'
$link = Join-Path $Folder 'Study Buddy.lnk'

function Show-Status {
    if (-not (Test-Path -LiteralPath $link)) {
        Write-Output "Not installed: there is no shortcut at $link"
        return
    }
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($link)
    Write-Output "Installed: $link"
    Write-Output "  runs:      $($shortcut.TargetPath) $($shortcut.Arguments)"
    Write-Output "  starts in: $($shortcut.WorkingDirectory)"
    if (-not (Test-Path -LiteralPath $shortcut.TargetPath)) {
        Write-Output "  WARNING: that file no longer exists. Run -Action add again from the current project folder."
    }
}

switch ($Action) {
    'add' {
        if (-not (Test-Path -LiteralPath $target)) {
            throw "start.bat was not found next to the scripts folder ($target). Run this from the Study Buddy project."
        }
        New-Item -ItemType Directory -Force -Path $Folder | Out-Null
        $shell = New-Object -ComObject WScript.Shell
        $shortcut = $shell.CreateShortcut($link)
        $shortcut.TargetPath = $target
        $shortcut.Arguments = 'wait'
        $shortcut.WorkingDirectory = $root
        $shortcut.WindowStyle = 7          # minimized
        $shortcut.Description = 'Starts Study Buddy when you sign in (waits for LM Studio first).'
        $shortcut.Save()
        Write-Output "Added the Startup shortcut."
        Show-Status
        Write-Output "To check it: press Win+R, type shell:startup, Enter. 'Study Buddy' should be in that folder. Sign out and in to try it."
    }
    'remove' {
        if (Test-Path -LiteralPath $link) {
            Remove-Item -LiteralPath $link
            Write-Output "Removed $link"
        } else {
            Write-Output "Nothing to remove: there is no shortcut at $link"
        }
    }
    'status' { Show-Status }
}
