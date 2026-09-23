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

## Skeleton (marker node)

`Shift+A > Rigs > Skeleton` gives you a marker container. A **marker** is a
world position (plus an optional orientation) with a stable key, drawn in the
3D viewport as a draggable handle in the `MRKS_rig` collection. The node holds
and shapes markers and nothing else: no mesh analysis, no auto placement, no
snapping.

A new node comes with **MediaPipe's 33 pose landmarks already loaded** (nose,
eyes, ears, mouth, shoulders, elbows, wrists, pinky / index / thumb, hips,
knees, ankles, heels, foot index) — the useful starting point for a body. They
are only a preset: rename them, delete the ones you do not want, add as many
of your own as the rig needs.

- **Add Marker** appends one at the 3D cursor, so it lands where you put it
  rather than piling up on the origin.
- **MediaPipe** reloads the preset. It adds only what is missing, unless you
  tick *Replace Markers* in the operator panel.
- **Clear Markers** (under *Advanced*) empties the node.
- Each marker has a name, a position, an **X** to delete it, and a gimbal
  button that switches it from position-only to position + rotation.

**Every marker gets its own output socket**, named after it. Drag from that
socket into a Custom Shape node's **Marker** input to place that bone — see
*Markers into Custom Shape* below. Renaming a marker renames its socket and
keeps the wire: links attach to the socket, not to its name.

1. **Show Markers / Front View**: creates one handle per marker (tinted with
   its MediaPipe side colour, or marker-pink for one you added) and switches
   the viewport to front orthographic. With **Lock Depth (2D)** on, handles
   only move in X/Z. With **Symmetric** on, right-side *MediaPipe* handles are
   locked and follow the left side mirrored across *Mirror X*; markers you
   added yourself have no mirror partner and are unaffected. Face, finger and
   toe landmarks move rigidly with their anchor (nose, wrist, ankle). Dragging
   a handle writes back into the node instantly.
2. **Rotation per marker**: every marker is position-only by default. The
   gimbal button switches one marker to position + rotation — its handle
   becomes an axis gizmo you can rotate, and an Euler field appears on the
   node. A marker's rotation supplies the **roll** of the bone placed there,
   and the **twist** when driving a rig (which two points alone can never
   recover — a forearm's twist, for instance).

### Markers into Custom Shape

The Custom Shape node has a **Marker** input. Wire a marker into it and that
bone is **posed** at the marker: dragging the handle moves the control exactly
as grabbing it in Pose mode would, and the custom shape follows because
Blender draws a widget at the posed bone.

Markers drive the pose **only**. The rest skeleton stays whatever the graph
built — a marker never edits Head/Tail, so it cannot deform the rig's
proportions or fight an Edit-mode change. Concretely:

- **Location** is always taken from the marker.
- **Rotation** only when that marker has rotation enabled (the gimbal button);
  otherwise the control keeps the orientation it had.
- **Scale** is never touched, so a scaled control stays scaled.

The node needs a bone name in its *Bone* field: without one there is no single
bone to pose, and nothing moves. While a marker is wired the node's *Pose*
Position/Rotation/Scale fields go read-only and are labelled **Pose (marker)**,
since the marker owns them and a typed edit would be overwritten on the next
rebuild. The *Rest* Head/Tail fields stay editable.

Posing runs as `marker_pose_pass()` at the end of every build, after the bones
exist and are in place — a bone's pose matrix resolves against its parent's
evaluated transform, so it cannot run earlier. The pass writes only when a
value actually differs, so it settles in one iteration instead of rewriting
the same matrix forever.

### The node has no bone output

Markers are the *only* output. Earlier versions also had a **Skeleton** socket
(a 22-bone body built from the landmark positions), a **Rig** socket (the same
thing as a pose, for retargeting onto a Rigify rig) and a **Parent** input
that re-rooted that skeleton. All three are gone, along with the entire
retarget subsystem they fed — the skeleton-to-Rigify match table, FK/IK drive
modes, bone overrides and `retarget.py`. Bones come from the Custom Shape
nodes the markers are wired into, and a marker poses its control directly, so
none of that indirection is needed.

Those sockets are stripped from older saved nodes on sight, along with their
links.

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
| `sockets.py` | Bone, Chain, Marker, Constraint sockets |
| `tree.py` | `ArmatureNodeTree` data-block, dirty tracking, optional live update |
| `nodes.py` | Bone, Chain, Mirror, Parent, Deform Group, Custom Shape, Skeleton, IK / generic constraint, Armature Output / Input nodes |
| `primary_rig.py` | Marker empties and locks, viewport overlay, MediaPipe landmark preset tables, marker operators |
| `widgets.py` | `WGTS_rig` widget library: finds/creates the hidden collection, lists `WGT-rig_*` meshes, generates Rigify-style presets (circle, cube, sphere, bone, diamond, root, gear, ...) |
| `build.py` | Forward compile: topological eval → edit-mode pass → pose-mode pass → marker pose pass, in-place rebuild by name, single undo step |
| `decompile.py` | Reverse: bone walk, chain-pattern collapsing, constraint node emission, depth-grid layout |
| `operators.py` | `armature_nodes.build`, `armature_nodes.decompile` |
| `ui.py` | Shift+A categories, header buttons, N-panel sidebar |

### Execution model notes

- **Marker posing** is a third pass at the end of the build: every Custom
  Shape node with a marker wired in poses its control. It runs last because a
  pose matrix resolves against the parent's *evaluated* transform, so the
  bones have to exist and be placed first. It writes only on a real
  difference, so it settles in one pass.
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
  `Custom Shape (one per widget group) -> Armature Output` with the output in
  **Custom Shapes Only** mode, bound to the rig by name. Building in that mode
  assigns shapes on the existing rig and leaves bones, constraints and drivers
  untouched.
  There is deliberately **no Armature Input node** in this graph. It used to
  feed the Custom Shape nodes and re-read the live rig on every evaluation,
  which made the armature -- not the graph -- the source of truth: node edits
  were overwritten on the next rebuild and the graph appeared frozen. Each
  Custom Shape node already stores its bone in full, so it drives the rig on
  its own. An Armature Input is still available by hand (Shift+A > Armature
  I/O) for the one case it suits: pulling an existing armature's bones INTO a
  graph that builds a different rig.
  The tradeoff: the Armature Input also held a whole-rig snapshot. Without it
  this graph stores the **control** bones only (each in its Custom Shape
  node), so if the rig object is deleted what rebuilds from the graph is the
  controls, not the `DEF-`/`MCH-`/`ORG-` layers. Use **Convert to Armature
  Nodes** (the full Bone/Chain graph) for a rig the nodes must be able to
  recreate whole.

## Deferred (per spec)

No MCP bridge, no continuous bidirectional sync, no anatomical validation.
