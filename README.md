# Armature Nodes — Procedural Armature Node System

A Blender addon that adds a node editor where a graph edits an armature, the
way Geometry Nodes edits a mesh. **Armature Input** puts the whole rig on the
wire, **Armature Output** writes it back, and every node in between is a
modifier: one value in, one value out, changing only what it selects.

## Install

1. Zip the `armature_nodes/` folder (the folder itself, so the zip contains
   `armature_nodes/__init__.py`).
2. In Blender: `Edit > Preferences > Add-ons > Install...`, pick the zip,
   enable **Armature Nodes**.

Requires Blender 3.6+ (tested API surface targets 3.6–4.x).

## Where it lives

Blender's Python API cannot register a brand-new *space type*, so — like
Sverchok and Animation Nodes — the addon registers a custom `NodeTree`.
Open any **Node Editor** and pick **Armature Nodes** from the header dropdown,
then click **New**. You get the standard node editor: Shift+A add menu, wire
dragging, N-panel sidebar, undo, saving with the .blend.

## The modifier model

Select an armature and open an Armature Nodes editor. The editor shows that
rig's graph, like the Shader Editor follows the active material. That graph is
**two nodes**:

```
Armature Input ──────────▶ Armature Output
   (the rig)                (write it back)
```

Not one node per bone. The rig *is* the input, so there is nothing to
reconstruct — exactly like opening Geometry Nodes on a mesh gives you Group
Input → Group Output and not a node per vertex.

To change something, drop a node onto the wire:

```
Armature Input ──▶ Position ──▶ Custom Shape ──▶ Armature Output
                   Bone: hand.L   Bone: hand.L
```

### One value on the wire

The **Rig** socket carries the *entire armature* — every bone, its rest
geometry, parenting, deform flags, constraints, widgets, and any pose the
graph has written so far. Not one bone: one rig.

- **Armature Input** emits the rig **unmodified**.
- Every node in between takes that whole rig in and returns a **modified
  copy**. On the Bone node the incoming rig arrives on the socket called
  *Parent* — the node upstream is what its bone hangs off — but it is the same
  whole-rig value every other node receives.
- **Armature Output** takes the last one and writes it back.

Every modifier node has a **Bone** field. Leave it empty and the node affects
every bone on the wire; name one (or several, semicolon separated) and it
affects only those. When the graph has a rig to look at, the field is a
searchable dropdown of that rig's real bone names.

The graph stays the size of your edits, not the size of the rig.

### The rig is the database

The Armature Output writes back to the same object the Input reads, so a live
read would feed the graph its own results. A stack that turns Deform off, or
replaces a widget, would see the changed value on the next evaluation and
could never get back to the original.

So the unmodified rig is stored **on the armature object itself**, in three
custom properties:

| Property | Holds |
| --- | --- |
| `an_rig_record` | the complete rig as JSON: bones, hierarchy, deform flags, collections, colours, display, pose settings, full constraint properties |
| `an_rig_widgets` | the widget **meshes**, zlib+base64 — so a rig can be rebuilt when its `WGT-*` objects are gone |
| `an_rig_touched` | which fields the last build wrote |

**Capture is explicit.** It happens when you click **Bind Rig**, and nowhere
else. Opening an editor does not capture, evaluating does not capture, and a
missing record is never filled in lazily — an unbound rig shows *Not bound*
with a Bind button and the graph writes nothing. That restriction is the whole
point: a capture is only correct when the rig is unmodified, and only a
deliberate click can promise that.

### Every build is restore-then-apply

```
base    = store.record.read(obj)          # the rig as it was
target  = graph evaluated over base       # what the graph asks for
changes = model.diff(base, target)        # per bone, per field
apply.pipeline(obj, base, target, store.touched.read(obj))
```

The pipeline does three things in order:

1. **Restore** every field the *last* build wrote that this one is not
   writing, back to its recorded value.
2. **Apply** this build's changes.
3. **Record** what it wrote, for the next build to restore.

Step 1 is why the viewport converges on `record + graph` whatever you do.
Deleting a node, unplugging the Input, or emptying the tree all leave paths in
the touched set that the next build puts back — so an empty graph means
**the rig equals its record**, not "leave whatever was there" and certainly
not a teardown. There is no special case for it, and there cannot be one.

Because the diff is per field, only what actually changed is written: two
identical builds write **zero** properties.

Only rest data is recorded. Pose transforms are not — posing is what the graph
does, and a record that remembered the live pose would fight the Transform
nodes on every build.

**`apply/` is the only writer.** Nothing else in the addon may assign to a
Bone, EditBone, PoseBone, Constraint or Armature property; a test walks the
AST and fails if anything does. (Marker handles are ordinary empties, not
armature data, and are written by the marker nodes.)


## Node categories

### Armature I/O

- **Armature Input** — nothing in, **Rig** out. Holds everything about the
  chosen armature: every bone's rest geometry, parenting, deform flags,
  constraints and existing custom shapes. It emits that rig **unmodified**.

  It does *not* read the live armature. The record lives **on the rig itself**
  (see *The rig is the database*), and this node inherits from it. If there is
  no record it emits nothing and shows *Not bound* with a Bind button.
- **Armature Output** — **Rig** in, nothing out. It is the display end of the
  graph: it writes the rig back *and* shows the markers of every Marker and
  Skeleton node feeding it, which its **Markers** toggle turns off for the
  whole graph at once. Leaving *Armature* blank targets the Input's source,
  which is the normal case. Two modes:
  - **Modify** (default) — writes shapes and pose onto the existing rig; its
    bones, constraints and drivers are left alone. Safe on a Rigify rig. The
    target is **exclusively the Armature Input's source**: the graph follows
    the binding, never the selection or a typed name.
  - **Full Rig** — the graph owns the armature and rebuilds its bones, so
    nodes can add and remove them.

### Marker

A marker is a world position with a stable key, drawn as a glowing sphere and
a draggable empty in the `MRKS_rig` collection. Placing things by dragging a
handle beats typing coordinates, which is the only reason markers exist.

Markers are displayed **through the Armature Output they feed**, the way a
value in Geometry Nodes only matters once it reaches the output. An
unconnected Marker node draws nothing; wire it in and it appears. Each node's
own **Handles** toggle still wins, for hiding one marker without unwiring it.

- **Marker** — one handle, as a position. Its **Position** output wires into a
  Bone node and dragging the handle moves that bone. It produces no bones and
  sits outside the Rig stream, the way a value node does in Geometry Nodes.

  A marker carries a position, a rotation and a scale, and its output is a
  **Transform** socket: all three on one wire. A Transform input (the
  Transform node's) takes all three; a Position, Rotation or Scale input
  takes the one that matches its type. A wired marker **is the field it
  replaces**: whatever the node would show and do with its own field, it
  shows and does with the marker. So wired to **one** bone — through a
  Position, Rotation, Transform or Bone node — it is live both ways, and the
  node says *Live: bone*:
  - **Wiring it in never moves the bone.** An absolute input (Position,
    Rotation, a Bone node's Position) puts the marker on the bone; a relative
    one (Transform's Translation / Rotation / Scale, an Offset) gives the
    marker the value the field held.
  - **Grab, rotate or scale the bone** in Pose mode and the marker and its
    handle follow.
  - **Drag, turn or scale the handle, or type a value**, and the bone follows.
  - **Unplug or delete it** and the field takes its value, so the bone stays.
  - Each value is labelled by the input it feeds (*Translation*, *Rotation*,
    *Scale*…). One that drives nothing is a greyed readout of the bone, and
    that part of the handle is locked.

  The handle always sits on the bone. For a relative input it is drawn where
  the offset puts the bone — rest plus the value, along world or the bone's
  own axes as the Transform node's Space says — rather than at the raw
  offset, which would leave it near the world origin.
- **Skeleton** — a bundle of markers, **one output socket each**: the
  Principled BSDF of markers. Each output is a position, so it wires into
  anything that takes one — a Position node, a Snap offset, a Custom Shape
  offset. It produces no bones itself.

  A new Skeleton node arrives with MediaPipe's 33 pose landmarks loaded (nose,
  eyes, ears, mouth, shoulders, elbows, wrists, pinky / index / thumb, hips,
  knees, ankles, heels, foot index) — a 1.8 m T-pose body to drag onto the
  character. They are a preset: rename them, delete what you do not want, add
  your own with **Add Marker** (which drops one at the 3D cursor).

  **Show Markers / Front View** creates the handles and switches to front
  orthographic. **Lock Depth (2D)** keeps handles in X/Z. **Symmetric** locks
  right-side MediaPipe landmarks and mirrors them from the left across
  *Mirror X*; markers you added have no mirror partner and are unaffected.
  Face, finger and toe landmarks move rigidly with their anchor.

  Every marker is position-only until its gimbal button is pressed, which
  turns the handle into a rotatable axis gizmo.

  Unlike a Marker node, a landmark is **not** moved onto the bone when you
  wire it in: the landmarks are a layout to drag onto a character, and the
  bones go to them. A landmark does follow when you grab the bone it drives.

### Bone

- **Bone** — **one** bone from the rig, posed. Pick any bone (DEF, MCH, ORG or
  a control — it makes no difference) and the node reads that bone's current
  world transform off the rig. Editing the values poses it.

  Pose only: rest geometry is never touched. Each of Location / Rotation /
  Scale has its own checkbox, so a node can move a bone without also pinning
  its rotation — an unchecked component is left exactly as the rig has it.

  It stays live after that: ticked values follow when you grab the bone,
  unticked ones are readouts (see *Live, both ways* under Transform).

  The incoming rig arrives on its **Parent** input. Its Constraints input is
  pose-stack data, so it only reaches the armature in **Full Rig** mode.
- **Chain** — generates N connected bones from a start, direction, length and
  an optional per-segment curve. Inputs Parent and Tip Constraints.

### Transform

All four write the **pose**, never rest geometry — so they cannot change the
proportions of a rig they are layered onto, and a custom shape follows because
Blender draws widgets at the posed bone. A freshly added node does nothing
until you ask it to, and a component the graph does not set keeps whatever the
rig already has.

- **Position** — like Geometry Nodes' *Set Position*.
  - **Position**: where the bones go, in world space. Used only when **Set
    Position** is ticked or something is wired in (a wired marker counts).
    Ticking the box first fills Position from the bone, so it never jumps.
  - **Offset**: added on top, along world axes.
- **Rotation** — the same pattern for orientation, in **degrees**: *Set
  Rotation* + Rotation for an absolute world orientation, Offset to turn on
  top. A wired marker supplies its own rotation.
- **Transform** — like *Transform Geometry*: Translation, Rotation and Scale,
  all relative, so several Transform nodes stack. The **Transform** input
  takes all three on one wire — wire a Marker into it and the handle's
  location, rotation and scale drive the bone together. While it is wired it
  replaces the three fields, which are hidden; unplug it and they come back
  holding its last values, so the bone stays put. **Space**:
  - **World** — along the scene axes, whatever way the bone points.
  - **Local** — along the bone's own axes. These *are* its Location and
    Rotation channels, the N-panel values, and they follow the parent the way
    hand-posing does.
- **Snap** — put the selected bones on a mesh. Snap To picks what "on" means:
  - **Origin** — the target object's own origin
  - **Bounding Box** — centre of its bounds
  - **Median** — mean of its vertices
  - **Volume** — volume centroid, the centre of mass of a solid. Unlike the
    median this ignores how densely the mesh is subdivided.
  - **Surface** — closest point on the surface to the bone

  plus an offset applied after the snap.

**Live, both ways.** A Position, Rotation, Transform or Bone node working on
a single bone mirrors it in real time:

- **Rig to node**: grab, rotate or scale the bone in Pose mode and the node's
  fields follow as you drag. If the value comes from a wired marker, the
  marker's handle moves with the bone too (see Marker, above).
- **Node to rig**: type a value and the bone moves. On Position and Rotation,
  typing into the field takes the bone over (ticks *Set*) — a field that
  looked like an input but only displayed was the old complaint.
- While a node is not driving a value (*Set* off, nothing wired, zero
  offset), that field is a plain readout of where the bone is.

The node tells its own writes apart from yours with a snapshot taken after
every build: anything that differs from it was you, and only that difference
is folded in. That is what keeps it from chasing its own output — which on a
constrained bone would oscillate for ever. Relative values (Offset, Transform)
measure your move from the rest pose, so moving the parent or the whole
object is not mistaken for a new offset. Undo, redo and file load reset the
snapshots. One live node per bone: two nodes linked to the same bone would
each pick up the same grab.

How the relative moves behave:

- **They are measured from the rest pose**, never from the live one. The live
  pose already contains the last build's move, so measuring from it would add
  the move again every build and the bone would drift.
- **Each bone moves once.** With a parent and its child both selected, a World
  move is applied to the parent and the child is carried — Blender's own
  G/R/S does exactly this. (Local is channel semantics, so it accumulates down
  a chain, as typing the same Location into every bone would.)
- **A later absolute value wins.** Set Position after a Transform replaces the
  Transform's offset, as in Geometry Nodes.
- **Blocked moves are reported.** A connected bone, a locked channel or a
  constraint can make Blender discard a pose; the Output and the sidebar say
  which, instead of reporting success.

### Shape

- **Custom Shape** — assigns a control widget from a Rigify-style preset, the
  `WGTS_rig` library, or any mesh object, with scale / rotation / wire width
  and a *Control Only* toggle that clears Deform. Its **Offset** input is a
  vector like Set Position: wire a marker into it and the widget follows the
  handle.

### Constraint

- **IK Constraint** and **Constraint** (copy/limit/track/stretch) — wired into
  the Constraints input of a Bone, Chain or Marker node.

## Execution model

0. **Baseline** — the Armature Input emits the rig's stored unmodified state,
   capturing it from the object on first use.
1. **Evaluate** — walk back from the Armature Output, memoized per node, into
   a list of `BoneDef`. Duplicate names are resolved last-write-wins so a
   downstream modifier beats an upstream one.
2. **Edit-mode pass** (Full Rig only) — create and place bones, set parenting.
3. **Pose-mode pass** — constraints and custom shapes.
4. **Pose-transform pass** — apply what the Transform nodes wrote. Runs last
   because a pose matrix resolves against the parent's *evaluated* transform;
   bones are handled parent-first with a view-layer flush per depth level, or
   children would inherit stale parents. It writes only on a real difference,
   so it settles in one pass instead of rewriting the same matrix forever.

Live Update re-runs this after a short debounce whenever a node value, link or
marker changes. Dragging a marker handle in the viewport writes back into its
node, which marks the tree dirty, which re-poses the rig.

## Architecture

| File | Responsibility |
| --- | --- |
| `core.py` | `BoneDef` / `ConstraintDef` / `ShapeDef`, eval context and memoization, bone selection |
| `model/` | Pure data: types, JSON schema, v1→v2 migration, diff, rig transforms. Imports no `bpy`. |
| `store/` | The `an_rig_*` properties, and the build lock. The only place they are touched. |
| `capture/` | Live armature → `RigRecord`. Never writes. Called only from Bind and Capture. |
| `apply/` | `RigRecord` → live armature. The only writer. |
| `bridge.py` | Seam between the old node graph's `BoneDef` and `RigRecord`, until `graph/` lands |
| `compat.py` | 3.6 / 4.x RNA shims (bone collections, colour, wire width) |
| `sockets.py` | Rig (the whole armature, the stream), Constraint, Vector sockets |
| `tree.py` | `ArmatureNodeTree` data-block, dirty tracking, live update |
| `nodes.py` | Every node type |
| `primary_rig.py` | Marker handles and locks, viewport overlay, MediaPipe preset table, marker operators |
| `widgets.py` | `WGTS_rig` widget library and Rigify-style preset generation |
| `build.py` | Forward compile: evaluate → edit pass → pose pass → pose-transform pass |
| `decompile.py` | Reverse: the two-node stack that targets a rig |
| `operators.py` | Build, decompile, convert |
| `ui.py` | Shift+A categories, header buttons, N-panel sidebar |
| `sync.py` | Editor follows the active armature; marker handles read back |

## Programmatic use

The tree is a normal data-block, so the whole system is scriptable with no UI:

```python
import bpy

tree = bpy.data.node_groups.new("Rig Nodes", "ArmatureNodeTreeType")
src = tree.nodes.new("ArmatureNodesInputNode")
src.source = bpy.data.objects["rig"]

move = tree.nodes.new("ArmatureNodesPositionNode")
move.bone = "hand.L"
move.inputs["Position"].default_value = (0.3, 0.0, 1.2)

out = tree.nodes.new("ArmatureNodesOutputNode")

tree.links.new(src.outputs["Rig"], move.inputs["Rig"])
tree.links.new(move.outputs["Rig"], out.inputs["Rig"])

bpy.ops.armature_nodes.build()
```
