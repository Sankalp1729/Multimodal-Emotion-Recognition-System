Param(
  [string]$ImageName = "mme-api:latest",
  [int]$Port = 8000,
  [string]$ModelsHostDir = "$PWD\multimodal_emotion_detection\models",
  [string]$TextAllowDownload = "false",
  [int]$Workers = 1
)

# Build the image
Write-Host "Building image $ImageName..."
docker build -t $ImageName .

# Ensure host models directory exists
if (!(Test-Path -Path $ModelsHostDir)) {
  Write-Host "Creating models directory at $ModelsHostDir"
  New-Item -ItemType Directory -Path $ModelsHostDir | Out-Null
}

# Run the container
Write-Host "Running container from $ImageName on port $Port with models mounted from $ModelsHostDir"
# Use ${} to delimit variables in strings containing ':' to avoid PowerShell parser issues
$portMapping = "${Port}:${Port}"
$volMapping = "${ModelsHostDir}:/models"

docker run --rm -it `
  -e MME_TEXT_ALLOW_DOWNLOAD=$TextAllowDownload `
  -e MME_MODELS_DIR=/models `
  -e PORT=$Port `
  -e UVICORN_WORKERS=$Workers `
  -p $portMapping `
  -v $volMapping `
  $ImageName