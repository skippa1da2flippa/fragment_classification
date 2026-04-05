# Download uv
1. *Windows*:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

2. *Linux/Mac*:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

# Pin uv python version
```bash
uv python pin <3.10.11>
```

# Create environment and install dependencies
```bash
uv sync
```
