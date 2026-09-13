# `render_canvas` implementation contract

Status: proposed; this document does not implement the tool.

## Purpose and scope

Let a connected Codex or other image-capable MCP client inspect the live banner
while choosing actions. Render the same environment instance used by
`get_canvas_state`, `execute_action`, and `get_current_reward`.

Keep implementation within the MCP adapter, shared rendering helpers, tests,
and usage documentation. Preserve reward formulas, TaskSpec interpretation,
transition semantics, and the Gymnasium observation space. No prompt parser,
model API, network transport, task browser, or file-export tool is required.

## Tool interface

Name: `render_canvas`

Description: “View the current canvas as a PNG image without changing the episode
or consuming an action. Available before and after episode completion.”

Input schema:

```json
{"type": "object", "properties": {}, "additionalProperties": false}
```

The client calls `render_canvas` with `{}`. Follow existing adapter conventions
for omitted arguments. Reject unknown arguments without changing the episode.
Accept no state, task, path, URL, scale, or styling overrides.

Successful result, shown schematically:

```json
{
  "content": [
    {"type": "image", "mimeType": "image/png", "data": "<base64 PNG bytes>"}
  ],
  "isError": false
}
```

Return a native MCP `ImageContent` inside `CallToolResult`. A file path or a
base64 string inside a text/JSON block is insufficient. Do not duplicate the
image in `structuredContent`. Omit `outputSchema` for this image-only tool;
preserve the existing JSON output schemas for the other tools.

Set `readOnlyHint=true`, `destructiveHint=false`, `idempotentHint=true`, and
`openWorldHint=false`. The current adapter identifies read-only tools by the
`get_` prefix; explicitly include `render_canvas` in that classification.

## Computation and invariants

Use this call order:

```text
validate arguments → acquire existing server lock → snapshot live state
                   → shared renderer → in-memory PNG → MCP image result
```

- Serialize the entire snapshot/render operation with existing server calls, so
  an action or reset cannot interleave with it. Never create a second environment
  or replay a trajectory to obtain the image.
- Reuse the existing drawing path in `rendering.py`. A small public PNG-bytes
  helper is acceptable; it must share rasterization with RGB rendering and PNG
  export. Encode using an in-memory buffer, without temporary files.
- The current renderer produces an RGB, 800×600 image. Preserve its clipping,
  stacking, fonts, text inset, shape labels, and image-placeholder behavior.
  Do not add overlays, resize, or improve the design while rendering.
- The MCP environment currently uses `render_mode=None`; calling `env.render()`
  alone returns `None`. Render its canonical snapshot through the shared renderer
  instead, without changing Gymnasium's render-mode contract.
- Rendering must not change canvas/target data, IDs, action count, lifecycle,
  reward issuance, or RNG state. It must not invoke reward evaluation.
- Allow empty, intermediate, and completed episodes. Rendering after `finish`
  must not reopen the episode or award its terminal reward again.
- Identical state produces identical PNG bytes under the locked software stack.
  Do not embed timestamps or other variable metadata. Cross-version Pillow/font
  byte identity is not promised.
- Use the existing tool-error convention for invalid arguments. Translate known
  renderer validation failures into `isError=true` with a concise `render_failed`
  code/message and no image. Do not mask unexpected programming errors as success.
  Keep stdout reserved for MCP protocol traffic.

## Acceptance checks

1. MCP discovery advertises the empty input schema and read-only annotations;
   the existing four tool schemas remain unchanged.
2. Through a real SDK stdio client, decode the returned image as a valid RGB PNG
   and compare its pixels with `render_rgb` of the same live canonical state.
3. Cover an empty canvas and a design containing text, a labeled shape, and an
   image placeholder, including clipping and overlapping z-order. Reuse renderer
   fixtures where appropriate.
4. Repeated reads return identical PNG bytes. Interspersing render calls in a
   legal action sequence preserves direct/MCP state, IDs, RNG, lifecycle, and
   reward parity; verify the rendering path does not call the reward evaluator.
5. A visible edit changes the image; resetting clears it; separate server
   instances remain isolated. Verify rendering shares action/reset serialization.
6. After `finish` and action-budget exhaustion, rendering still succeeds and
   further edits still require reset. Invalid arguments and simulated known
   renderer failures leave the episode unchanged.
7. Run Ruff and the complete pytest suite. Then perform a Codex smoke test that
   calls the tool, inspects the image, makes a visible edit, and renders again.
   Report protocol/pixel verification separately from actual model image access;
   if the latter cannot be tested, state that explicitly.

## Demo and documentation

Update `docs/MCP.md` with the tool and Codex connection instructions once it is
implemented. Demonstrate an LLM choosing its own actions:

```text
reset → read TaskSpec/state → edit → render → inspect reward → revise → finish
```

Render again after finishing to inspect the final banner. Reference trajectories
may support automated tests, but must not supply the interactive demo's actions.
Rendering is an observation, not a new quality metric: existing reward
approximations and limitations remain unchanged. Image display and model image
access depend on the connected client's capabilities.
