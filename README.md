# Armature Nodes — Procedural Armature Node System

A Blender addon that adds a custom, wireable node editor where a node graph
compiles into a complete working armature (forward), and an existing armature
decompiles into the equivalent node graph (reverse). The node tree is a real
Blender data-block — buildable by hand in the GUI or fully programmatically.

## Install

1. Zip the `armature_nodes/` folder (the folder itself, so the zip contains
   `armature_nodes/__init__.py`).
2. In Blender: `Edit > Preferences > Add-ons > Install...`, pick the zip,
   enable **Armature Nodes**.

Requires Blender 3.6+ (tested API surface targets 3.6–4.x).

## Where it lives

Blender's Python API cannot register a brand-new *space type*, so — like
Sverchok and Animation Nodes — the addon registers a custom `NodeTree`.
Open any **Node Editor** area and pick the **Armature Nodes** tree type from
the editor header dropdown (armature icon), then click **New** to create a
tree. You get the standard node editor interaction model: Shift+A add menu
with categories (Bones, Chains, Constraints, Armature I/O), wire dragging,
N-panel sidebar, undo, saving with the .blend.

## Live rig workflow

1. Select an armature in the 3D viewport and open an **Armature Nodes** editor.
   The editor automatically displays that rig's bound node tree, like the
   Shader Editor follows the active material.
2. Edit bone, constraint, or widget values on the nodes. With **Live Update**
   enabled, changes are applied to the real armature automatically after the
   short debounce interval.
3. Use **Refresh From Rig** only when you want to replace the node graph with
   a fresh decompile of the selected rig. The graph stores snapshots of bones,
   constraints, and widget meshes, so it can rebuild the rig if the original
   object is deleted.

## Primary Rig (MediaPipe marker skeleton)

`Shift+A > Rigs > Primary Rig` gives you the 33 MediaPipe Pose landmarks
(nose, eyes, ears, mouth, shoulders, elbows, wrists, pinky / index / thumb,
hips, knees, ankles, heels, foot index), drawn live in the 3D viewport as the
MediaPipe skeleton — orange left side, cyan right side, grey links. The node
does nothing but hold and shape those markers; there is no mesh analysis, no
auto placement and no snapping.

1. **Show Landmarks / Front View**: one draggable handle per landmark (in the
   `MRKS_rig` collection, tinted with its side colour) and the viewport
   switches to front orthographic. With **Lock Depth (2D)** on, the handles
   only move in X/Z. With **Symmetric** on, right-side handles are locked and
   follow the left side mirrored across *Mirror X*. Face, finger and toe
   landmarks move rigidly with their anchor (nose, wrist, ankle). Dragging a
   handle writes back into the node instantly.
2. **Rotation per landmark**: every landmark is position-only by default. The
   gimbal button on a landmark row switches that one landmark to
   position + rotation — its handle becomes an axis gizmo you can rotate, and
   an Euler field appears on the node. Any of the 33 can be enabled
   individually. A landmark's rotation supplies the **roll** of the bone that
   starts there, and the **twist** when driving a rig (which two points alone
   can never recover — a forearm's twist, for instance).

### The two outputs

- **Skeleton** → wire into an **Armature Output**. The marker skeleton as-is:
  22 bones (hips, 3 spine bones, neck, head, shoulders, arms, hands, thighs,
  shins, feet, toes), no rig matching involved. The optional *Parent* input
  re-roots the whole skeleton under an existing bone.
- **Rig** → wire into the **Skeleton** input of an **Armature Input** node
  that points at a generated Rigify rig. The skeleton's bones are matched to
  that rig's controls by name and the controls are posed to follow the
  markers. This happens automatically on every update once the link exists —
  there is no button. Bone names are Blender/Rigify metarig style
  (`upper_arm.L`, `thigh.R`).

### Bone matching (skeleton → Rigify rig)

The match targets **control bones**, not `DEF-`/`ORG-`. In a generated rig
every `DEF-` bone carries a Copy Transforms / Stretch To constraint from
`ORG-`/`MCH-`, and `ORG-` bones are themselves constrained to the controls, so
a pose written into either layer is overwritten on the next depsgraph
evaluation. Controls are the only unconstrained, writable layer. `ORG-` and
`DEF-` names are kept only as fallbacks, for hand-made or non-Rigify rigs
where the control names are absent.

FK is the default drive mode: one FK control per skeleton bone, unambiguous.
IK is opt-in on the Armature Input node; there the hands and feet drive the
`*_ik` controls and the elbow / knee markers place the pole targets. Either
way the node can flip Rigify's IK/FK slider so the driven chain is the one
the rig follows.

| Skeleton bone | Control (first match wins) | Fallbacks |
| --- | --- | --- |
| `hips` | `torso` (location + rotation) | `hips`, `ORG-spine`, `DEF-spine` |
| `spine` | `tweak_spine.001` | `spine_fk.001`, `ORG-spine.001`, `DEF-spine.001` |
| `spine.001` | `tweak_spine.002` | `spine_fk.002`, `ORG-spine.002`, `DEF-spine.002` |
| `spine.002` | `chest` | `tweak_spine.003`, `ORG-spine.003`, `DEF-spine.003` |
| `neck` | `neck` | `tweak_spine.004`, `ORG-spine.004`, `DEF-spine.004` |
| `head` | `head` | `tweak_spine.005`, `ORG-spine.005`, `DEF-spine.005` |
| `shoulder.{L,R}` | `shoulder.{S}` | `ORG-shoulder.{S}`, `DEF-shoulder.{S}` |
| `upper_arm.{L,R}` | `upper_arm_fk.{S}` *(IK: `upper_arm_ik.{S}`)* | `ORG-upper_arm.{S}`, `DEF-upper_arm.{S}` |
| `forearm.{L,R}` | `forearm_fk.{S}` *(IK: pole `upper_arm_ik_target.{S}`)* | `ORG-forearm.{S}`, `DEF-forearm.{S}` |
| `hand.{L,R}` | `hand_fk.{S}` *(IK: `hand_ik.{S}`, location + rotation)* | `ORG-hand.{S}`, `DEF-hand.{S}` |
| `thigh.{L,R}` | `thigh_fk.{S}` *(IK: `thigh_ik.{S}`)* | `ORG-thigh.{S}`, `DEF-thigh.{S}` |
| `shin.{L,R}` | `shin_fk.{S}` *(IK: pole `thigh_ik_target.{S}`)* | `ORG-shin.{S}`, `DEF-shin.{S}` |
| `foot.{L,R}` | `foot_fk.{S}` *(IK: `foot_ik.{S}`)* | `ORG-foot.{S}`, `DEF-foot.{S}` |
| `toe.{L,R}` | `toe_fk.{S}` | `toe.{S}` (pre-0.6.3 rigs), `ORG-toe.{S}`, `DEF-toe.{S}` |
| `thumb.{L,R}` (optional) | `thumb.01.{S}` | `ORG-thumb.01.{S}`, `DEF-thumb.01.{S}` |

Never driven: `root`, `foot_heel_ik.*`, the finger controls
(`f_index.*`, `f_middle.*`, `f_ring.*`, `f_pinky.*`, `palm.*`) and face
controls. Three MediaPipe points per hand cannot solve per-phalanx rotations,
so fingers stay with the animator; add them by hand through the node's
override map if you need them.

Only `hips` (and, in IK mode, the hand / foot / pole controls) drives
*location* — everything else is rotation-only, so the rig keeps its own
proportions instead of being stretched onto the markers. Control scale is
preserved. Poses are applied parent-before-child with a view-layer flush per
hierarchy level, which Blender requires or children inherit stale parents.

The Armature Input node reports `22/22 bones` and lists any skeleton bone it
could not match, so a rig with non-standard names tells you exactly what to
override.

**Manual overrides.** Any pairing can be re-pointed, not just the failures:
the *Bone Overrides* list on the Armature Input node takes a skeleton bone and
a control bone (searchable against the rig's actual bone list) and wins over
the table for that bone. *Fill From Matches* writes out a row per resolved
bone so you can edit the whole mapping, *Add* starts an empty row, and each
unmatched bone in the report has a `+` that pre-fills a row for it. A row can
be disabled with its checkbox without deleting it. An override naming a bone
the rig does not have is reported as unmatched rather than quietly falling
back to the automatic choice — a manual mapping is a decision, not a hint.

## Programmatic use (zero UI)

```python
import bpy

tree = bpy.data.node_groups.new("MyRig", "ArmatureNodeTreeType")
chain = tree.nodes.new("ArmatureNodesChainNode")
chain.prefix, chain.count, chain.bone_length = "spine", 5, 0.3
out = tree.nodes.new("ArmatureNodesOutputNode")
out.armature_name = "SpineRig"
tree.links.new(chain.outputs["Chain"], out.inputs["Bones"])

from armature_nodes.build import build_armature_from_tree
obj = build_armature_from_tree(tree)   # or: bpy.ops.armature_nodes.build(tree_name="MyRig")
```

## Architecture

| File | Contents |
| --- | --- |
| `core.py` | `BoneDef` / `ConstraintDef` intermediate model, eval context, memoization |
| `sockets.py` | Bone, Chain, Armature, Pose, Constraint, Transform, Float, Vector, Bool sockets |
| `tree.py` | `ArmatureNodeTree` data-block, dirty tracking, optional live update |
| `nodes.py` | Bone, Chain, Mirror, Parent, Deform Group, Custom Shape, Primary Rig, IK / generic constraint, Armature Output / Input nodes |
| `primary_rig.py` | MediaPipe landmark tables, per-landmark position/rotation handling, marker empties, viewport overlay, skeleton generation |
| `retarget.py` | Skeleton bone -> Rigify control mapping table, name matching with fallbacks, pose application (FK / IK) |
| `widgets.py` | `WGTS_rig` widget library: finds/creates the hidden collection, lists `WGT-rig_*` meshes, generates Rigify-style presets (circle, cube, sphere, bone, diamond, root, gear, ...) |
| `build.py` | Forward compile: topological eval → edit-mode pass → pose-mode pass, in-place rebuild by name, single undo step |
| `decompile.py` | Reverse: bone walk, chain-pattern collapsing, constraint node emission, depth-grid layout |
| `operators.py` | `armature_nodes.build`, `armature_nodes.decompile` |
| `ui.py` | Shift+A categories, header buttons, N-panel sidebar |

### Execution model notes

- **Retarget** is a third pass at the end of the build: every Armature
  Input node with something wired into its Skeleton input drives its rig's
  controls. A graph that *only* retargets (Primary Rig -> Armature Input,
  no Armature Output) is valid and skips the compile entirely.
- **Forward** is two-pass because Blender requires it: edit bones first,
  pose-bone constraints second. Rebuilds match by bone name and update the
  existing armature object in place instead of duplicating.
- **Live update** is on by default; disable it per-tree in the sidebar. It
  defers rebuilds through `bpy.app.timers` because the tree `update()`
  callback runs in a restricted context.
- **Every editable value reaches the rebuild, not just node properties.**
  Most bone/chain/constraint values in this system are set the Shader-Editor
  way: as the default_value of an unconnected input socket, not a property on
  the node itself. `nodes.py` injects a rebuild-triggering `update` callback
  into every node property at register time, but a `NodeSocket` is a
  different class hierarchy that injection never reaches -- so
  `FloatSocket`/`VectorSocket`/`BoolSocket`'s `default_value` carry their own
  `update` callback (`sockets.py`), and typing into any unconnected socket
  rebuilds exactly like changing a property on the node does.
- **Deletion is a change like any other.** Every node's `free()` schedules a
  rebuild, because `NodeTree.update()` is not a dependable deletion signal; a
  4 Hz topology watcher in `sync.py` catches removals done from Python. A
  graph that no longer produces bones tears its armature down instead of
  freezing it, and an armature the graph *generated* (tagged `an_owner_tree` /
  `an_owner_node` at creation) is removed with the node that owned it.
  Armatures the graph did not create are never deleted. Removal is queued and
  run on the debounce timer, since `free()` is too restricted a context to
  delete data-blocks in.
- **Reverse** is a one-shot operator into a *new* tree (never clobbers an
  existing graph). Straight, evenly spaced, connected runs of 3+ bones with
  no mid-run constraints collapse into a single Chain node; everything else
  becomes individual Bone nodes. Constraints on a chain tip become nodes
  wired into `Tip Constraints`.
- **Custom shapes** (Shift+A > Bones > Custom Shape) mark bones as controls.
  Wire any bone/chain output through it. Three sources: **Preset** generates a
  per-bone `WGT-rig_<bone>` wire mesh into the `WGTS_rig` collection (excluded
  from the view layer, exactly like Rigify); **WGTS_rig** picks one existing
  widget from that collection for all incoming bones; **Object** uses any
  mesh. Scale / translation / rotation, wire width, and "scale to bone
  length" map 1:1 onto Blender's pose-bone custom-shape settings, and
  *Control Only* switches Deform off. Decompile reads `custom_shape` back
  and groups bones sharing a widget + transform into a single node; each
  node's *Bones* filter lists exactly which bones receive the shape.
  The **Position / Scale** fields at the top of a node are the controlled
  bone's **world-space** transform (armature object transform x current pose,
  measured from the world origin, not the bone's own origin). They are live:
  grabbing, posing or animating the bone in the viewport updates the node,
  and typing a value moves/scales the bone to match. The widget's own
  bone-relative *Offset / Widget Scale / Rotation* live in the Widget section.
- **Rigs** are any armature that has control bones: bones carrying a
  `custom_shape` that live in a visible bone collection / layer. This is not
  tied to Rigify; a Rigify-generated rig qualifies, a metarig (no custom
  shapes) does not, and a hand-made rig with shaped controls qualifies too.
  Decompiling a rig emits **no Bone/Chain nodes at all**: only
  `Armature Input(rig) -> Custom Shape (one per widget group) -> Armature
  Output` with the output in **Custom Shapes Only** mode. Building in that
  mode assigns shapes on the existing rig and leaves bones, constraints and
  drivers untouched.

## Deferred (per spec)

No MCP bridge, no continuous bidirectional sync, no anatomical validation.
