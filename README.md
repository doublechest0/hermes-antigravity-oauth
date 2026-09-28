# hermes-antigravity-oauth

Run [Hermes Agent](https://hermes-agent.nousresearch.com) on your Google Antigravity plan (Gemini 3.x, Claude 4.6, GPT-OSS) with a single sign-in.

```bash
npx hermes-antigravity-oauth          # install + enable the plugin, then sign in
hermes --provider antigravity-oauth -m claude-sonnet-4-6
```

## How it works

- **The calls are Antigravity CLI calls.** Every model request is made by Google's official `agy` binary, using agy's own login. The plugin sends no HTTP requests of its own and never imitates Google's OAuth client.
- **Sign-in belongs to agy.** `hermes auth add antigravity-oauth` runs agy's browser OAuth flow and then confirms it live with `agy models`. Over SSH, paste the code back when asked. The token stays wherever agy stores it (OS keychain or agy's token file). The plugin never reads, copies, uploads, or writes it.
- **What it does touch (disclosure):** each agy process runs in a private temp HOME. To let agy reuse its own login there, the plugin symlinks agy's `antigravity-oauth-token` file into that HOME, and on macOS it symlinks `~/Library/Keychains`. Before spawning, it checks the OS keyring for the credential's existence only (`security` / `secret-tool` / `cmdkey`) and never reads the secret. The temp HOME is deleted when the session closes.
- **Your Hermes SOUL.md is the persona.** Each turn, the plugin writes Hermes' system prompt into agy's private workspace as `GEMINI.md`, which agy loads as first-class rules. That prompt starts with `$HERMES_HOME/SOUL.md`, and the file is added directly if Hermes left it out.
- **Hermes runs the tools.** agy's built-in tools never execute on your host. When a model calls one anyway, the plugin re-issues it as the matching Hermes tool: `run_command`→`terminal`, `view_file`→`read_file`, `grep_search`/`find_by_name`→`search_files`, `write_to_file`→`write_file`, `search_web`→`web_search`, `read_url_content`→`web_extract`. It then runs under Hermes' approvals and sandbox. Any other native tool is blocked.

## Install

Requirements: Hermes Agent, plus the Antigravity CLI (`agy`) from <https://antigravity.google/cli>.

| Route | Command |
|---|---|
| npx (recommended) | `npx hermes-antigravity-oauth` (add `--yes` to install agy if it is missing) |
| Hermes directly | `hermes plugins install https://github.com/neerazz/hermes-antigravity-oauth --enable` then `hermes auth add antigravity-oauth` |

npx commands: `install` (default), `login`, `status`, `uninstall`. Flags: `--yes`, `--no-login`, `--force`, and a 40-hex commit SHA to pin the version.

## Auth commands

| Command | Effect |
|---|---|
| `hermes auth add antigravity-oauth` | Runs agy's Google sign-in, then verifies it live |
| `hermes auth status antigravity-oauth` | Live check through `agy models` |
| `hermes auth logout antigravity-oauth` | Explains agy's `/logout` (Hermes holds no token) |

Aliases: `google-antigravity`, `agy-oauth`.

## Configuration

| Env var | Default | Purpose |
|---|---|---|
| `ANTIGRAVITY_COMMAND` | `agy` | Path to the agy binary |
| `ANTIGRAVITY_WORKSPACE_RULES` | `1` | `0` puts SOUL/system prompt inline in the prompt instead of `GEMINI.md` |

Claude and GPT-OSS models get no `--effort` flag because agy rejects it for them. Gemini models accept the `-low/-medium/-high` suffixes.

## Development

```bash
PYTHONPATH=<hermes-agent checkout>:. python tests/test_auth_handler.py   # etc. for each tests/test_*.py
```

## Releases

Every push to `main` runs `.github/workflows/release.yml`. It first runs the full test suite against Hermes Agent `main`. Then it publishes the next version to npm with provenance, using npm trusted publishing (GitHub OIDC), so the repo stores no npm token. Finally it tags the commit `vX.Y.Z`.

- Default: patch bump from the latest published version.
- Minor/major: set `version` in `package.json` above the published one in the commit, and that exact version is published.
- A red test run publishes nothing.

## Credits

The streaming client, process isolation, and keyring probes come from [soyelmismo/hermes-antigravity-subscription](https://github.com/soyelmismo/hermes-antigravity-subscription) (MIT) at `b7ab470`. This package adds the auth handler, SOUL delivery, native-tool bridging, and the npx installer.

MIT © Neeraj Kumar Singh Beshane
