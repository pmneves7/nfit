#ifndef AppVersion
  #error AppVersion is required
#endif
[Setup]
AppId=org.nfit.desktop
AppName=nfit
AppVersion={#AppVersion}
AppPublisher=Paul M. Neves
DefaultDirName={localappdata}\Programs\nfit
DefaultGroupName=nfit
UninstallDisplayIcon={app}\nfit.exe
SetupIconFile={#IconFile}
OutputDir={#OutputDir}
OutputBaseFilename={#InstallerName}
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
CloseApplications=yes
RestartApplications=no
[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
[Icons]
Name: "{autoprograms}\nfit"; Filename: "{app}\nfit.exe"
Name: "{autodesktop}\nfit"; Filename: "{app}\nfit.exe"; Tasks: desktopicon
[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked
