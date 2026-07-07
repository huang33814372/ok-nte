$py = if (Test-Path .\.venv\Scripts\python.exe) { ".\.venv\Scripts\python.exe" } else { "python" }

& $py -m unittest discover -s tests -p "*test*.py" --top-level-directory . -v

if ($LASTEXITCODE -ne 0) {
    throw "Tests failed"
}
