# API compatibility rules

The server must be a drop-in replacement for TypeSafe's System One API.

- **The OpenAPI snapshot in `schemas/` is the source of truth** for request/response shapes. Never edit
  it by hand, and never hand-edit generated models; regenerate them from the snapshot.
- Refreshing the snapshot is a deliberate change: download the new spec as a new dated file, regenerate,
  and note the diff in the commit message.
- **No extra fields in JSON bodies**, and no renamed or reshaped fields. Anything we need beyond Jev's
  schema (timings, label mass, prompt hash) goes in traces or response headers, never the body.
- Status codes and error bodies follow `api-schema.md` (401 / 422 / 429 / 529 / 500).
- Declared deviations are listed in `api-schema.md` (currently: the per-profile option cap). Adding one
  requires updating that list.
- **Contract tests use the official `typesafe-sdk`** pointed at our server. A change that breaks them is a
  compatibility bug, even if our own tests pass.
- Question keys are never sent to the model.
