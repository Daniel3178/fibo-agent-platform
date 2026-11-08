---
name: git-conventions
description: Branching and commit conventions for the sebx workspace repos (fibonacci_neural_sde, sebx-agent-platform). Use before making any code change, creating a branch, staging files, or writing a commit in these repos — covers branching off the default branch, Conventional Commit messages (feat:, fix:, chore:, ...), and the 200-line-per-commit limit.
---

# Git conventions for the sebx workspace

Apply these rules in every git repo of this workspace (e.g. `../fibonacci_neural_sde`, whose default branch is `main`).

## 1. Branch off the default branch before changing code

Never commit on the default branch, and never push to it directly. Before editing code:

```bash
default=$(git symbolic-ref --short refs/remotes/origin/HEAD 2>/dev/null | sed 's#^origin/##')
default=${default:-main}
git branch --show-current          # if this equals $default, create a branch:
git fetch origin
git switch -c <type>/<short-kebab-description> "origin/$default"
```

If the working tree already has uncommitted changes on the default branch, `git switch -c <branch>` (without a start point) carries them onto the new branch.

If you are already on a feature branch for the same task, keep working there.

## 2. Branch names

`<type>/<short-kebab-description>`, using the same types as commit messages:

- `feat/temperature-annealing`
- `fix/gating-nan-loss`
- `docs/gating-diagnostics-readme`
- `refactor/expert-router`

Lower-case, hyphen-separated, a few words describing the change.

## 3. Commit messages (Conventional Commits)

```
<type>(<optional scope>): <summary>

<optional body: why the change was made, not how>

<optional footer, e.g. BREAKING CHANGE: ...>
```

Summary rules: imperative mood ("add", not "added"), lower-case, no trailing period, at most 72 characters.

| Type       | Use for                                                        |
|------------|----------------------------------------------------------------|
| `feat`     | a new feature or capability                                    |
| `fix`      | a bug fix                                                      |
| `chore`    | maintenance that touches no source behaviour (deps, configs)   |
| `docs`     | documentation only                                             |
| `refactor` | code restructuring with no behaviour change                    |
| `test`     | adding or fixing tests                                         |
| `perf`     | performance improvements                                       |
| `style`    | formatting, whitespace, naming — no logic change               |
| `build`    | build system or packaging (`pyproject.toml`, etc.)             |
| `ci`       | CI configuration                                               |
| `revert`   | reverting an earlier commit                                    |

Breaking changes: add `!` after the type (`feat!: ...`) or a `BREAKING CHANGE:` footer.

Examples:

- Good: `feat(gating): add temperature annealing to expert router`
- Good: `fix: prevent nan loss when entropy weight is zero`
- Good: `docs: describe gating diagnostics in readme`
- Bad: `Added Gating Diagnostics (Experiment 1 and 2)` — no type, past tense
- Bad: `feat: add regional variance, temperature annealing, adaptive entropy and ...` — several changes in one commit; split them

## 4. At most 200 changed lines per commit

A commit may change at most 200 lines (insertions + deletions, binary files ignored). Before every commit, stage the files, then run:

```bash
bash .claude/skills/git-conventions/scripts/check-commit-size.sh
```

Run it from the repo being committed to, using the script's absolute path when that repo is not the workspace root (e.g. `bash <workspace>/.claude/skills/git-conventions/scripts/check-commit-size.sh`).

If it fails, split the work into several commits, each a meaningful unit on its own:

```bash
git restore --staged .
git add <files for the first logical change>   # or: git add -p <file>
bash <path>/check-commit-size.sh
git commit -m "<type>: <summary>"
# repeat for the remaining changes
```

Plan large features as a sequence of small commits from the start (e.g. `refactor:` groundwork, then `feat:` core logic, then `test:`, then `docs:`).

If one change cannot be split below the limit (a generated file, a dataset, a large rename), stop and ask the user before committing.

## 5. Checklist

1. On the default branch? Create `<type>/<description>` from `origin/<default>`.
2. Make the change in small, logical steps.
3. Stage one logical step and run `check-commit-size.sh`.
4. Commit with a Conventional Commit message.
5. Push or open a PR only when the user asks.
