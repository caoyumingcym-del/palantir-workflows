# ica_tools

Automates (re-)importing this pipeline into ICA at the current git commit,
via the same API `Import from Git` in the ICA UI uses -- because that UI
doesn't let you edit a pipeline's commit after the first import, and this
pipeline has gone through several rounds of fixes already. Adapted from
`SingleCell/PIPseqDownsample/ica_tools/export_pipeline_to_ica.py` elsewhere
in this repo; see that file/its own docs for the pattern this is based on.

## One-time setup

1. **API key**: generate one in the ICA web UI (user/profile settings ->
   API keys) and save just the key string to `~/.icav2/api_key.txt`.
   Never commit this file or paste the key anywhere else.
2. **Project ID**: open `ica_common.py` and add your ICA project's name and
   ID to `PROJECT_NAMES_AND_IDS`. Find the ID in the project's URL in the
   ICA web UI, or via `icav2 projects list` if you have the `icav2` CLI set
   up.
3. **Git credential**: leave `GIT_CREDENTIAL_UUID` as `''` -- per Illumina's
   own docs, this is only needed for private repositories, and
   `caoyumingcym-del/palantir-workflows` is confirmed public (fetching a
   file from it needs no authentication at all). If the import ever fails
   complaining about repository access, create one at **System Settings >
   Credentials > Create > Git Credential** (a GitHub Personal Access Token)
   and paste its ID into `GIT_CREDENTIAL_UUID`.

The API key deliberately lives outside the repo entirely -- never commit
`~/.icav2/api_key.txt` or paste its contents anywhere. `PROJECT_NAMES_AND_IDS`
is a placeholder until you fill it in; it isn't a secret (your colleague's
own script commits real project IDs the same way), but decide deliberately
whether committing yours is meant for just you or for whoever else pulls
this branch.

## Usage

```bash
cd PIPSeq_QCreport/ica_tools
python3 export_pipeline_to_ica.py
```

Prompts you to pick a project, then:

1. Calls ICA's `importGitPipeline` at the current `git rev-parse HEAD`,
   creating a **new** pipeline entry named `PIPSeq_QCreport_<short-commit>`
   (every run makes a new entry rather than updating one in place -- you'll
   accumulate one per export; delete old ones in the ICA UI as you like).
2. Polls until that pipeline reaches `Draft` status.
3. Uploads `../inputForm.json` to it.

Run it again after every commit you want ICA to actually pick up --
pushing to the branch alone does not update anything already imported.
