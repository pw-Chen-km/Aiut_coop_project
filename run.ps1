$env:QA_BUNDLE_DIR = (Resolve-Path "$PSScriptRoot/skill-bundle")
$env:QA_CONFIG = (Resolve-Path "$PSScriptRoot/config/qa_agent.toml")
python "$PSScriptRoot/service/app.py"
