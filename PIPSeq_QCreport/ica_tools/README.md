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
3. **Git credential**: open `export_pipeline_to_ica.py` and set
   `GIT_CREDENTIAL_UUID` to an ICA git credential authorized to read
   `https://github.com/caoyumingcym-del/palantir-workflows` (find/create one
   under this project's credentials in the ICA UI). If that repository is
   public, try leaving it as an empty string first -- a credential may not
   be required at all for a public repo.

None of the above are committed as filled-in values (the API key
deliberately lives outside the repo entirely; the project ID and git
credential UUID are placeholders in these files until you fill them in
locally) -- fill them in on your own machine, don't push real values back
into `PROJECT_NAMES_AND_IDS`/`GIT_CREDENTIAL_UUID` unless you're sure that's
intended for whoever else pulls this branch.

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
