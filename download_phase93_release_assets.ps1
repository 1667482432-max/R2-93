$ErrorActionPreference = 'Stop'
$repo = '1667482432-max/R2-93'
$tag = 'phase93-complete-project'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pointer = Get-Content (Join-Path $root 'phase93_release_assets.json') -Raw | ConvertFrom-Json
$api = "https://api.github.com/repos/$repo/releases/tags/$tag"
$release = Invoke-RestMethod -Uri $api -Headers @{Accept='application/vnd.github+json';'User-Agent'='phase93-client'}
$byName = @{}
foreach($asset in $release.assets){ $byName[$asset.name] = $asset }
foreach($item in $pointer.assets){
  if(-not $byName.ContainsKey($item.name)){ throw "Missing release asset: $($item.name)" }
  $dest = Join-Path $root $item.name
  if($item.name -eq 'Round2_Test_Channel_phase40_p9_actual_antip10_pas050.npy'){
    $dest = Join-Path $env:TEMP $item.name
  }
  Invoke-WebRequest -Uri $byName[$item.name].browser_download_url -OutFile $dest
  $actual = (Get-FileHash -Algorithm SHA256 -LiteralPath $dest).Hash.ToLower()
  if($actual -ne $item.sha256){ throw "SHA256 mismatch for $($item.name): $actual" }
}
Write-Output "Phase93 assets ready under $root"
