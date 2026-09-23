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

### The rig stores its own baseline

The Armature Output writes back to the same object the Input reads, so a live
read would feed the graph its own results. A stack that turns Deform off, or
replaces a widget, would see the changed value on the next evaluation and
could never get back to the original.

So the unmodified rig is serialised once and stored **on the armature object**
as a custom property (`an_baseline`), the first time a graph is bound to it.
Every evaluation starts from that same base state, which is what makes the
stack behave like modifiers:

- Deleting a node genuinely undoes it, instead of leaving its effect baked in.
- Unplugging the Armature Input cannot lose anything — the record is on the
  rig, not in the wire.
- An empty graph is **not** a teardown. It means "no modifications defined",
  so the rig is left exactly alone. (It used to strip every widget, which
  leaves a Rigify rig looking precisely like a metarig, and no replug could
  undo it.)

Only rest data is stored — bones, hierarchy, deform flags, constraints and
custom shapes. Pose is deliberately excluded: posing is what the graph does,
and a baseline that remembered it would fight the Transform nodes.

The refresh button on the Armature Input re-captures, for when you have edited
the armature itself. It asks first, because it captures whatever the rig looks
like *now* — including anything the graph has already applied.

## Node categories

### Armature I/O

- **Armature Input** — nothing in, **Rig** out. Holds everything about the
  chosen armature: every bone's rest geometry, parenting, deform flags,
  constraints and existing custom shapes. It emits that rig **unmodified**.

  It does *not* read the live armature. The record lives **on the rig itself**
  (see *The rig stores its own baseline*), and this node inherits from it.
- **Armature Output** — **Rig** in, nothing out. It is the display end of the
  graph: it writes the rig back *and* shows the markers of every Marker and
  Skeleton node feeding it, which its **Markers** toggle turns off for the
  whole graph at once. Leaving *Armature* blank targets the Input's source,
  which is the normal case. Two modes:
  - **Modify** (default) — writes shapes and pose onto the existing rig; its
    bones, constraints and drivers are left alone. Safe on a Rigify rig.
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

### Bone

- **Bone** — **one** bone from the rig, posed. Pick any bone (DEF, MCH, ORG or
  a control — it makes no difference) and the node reads that bone's current
  world transform off the rig. Editing the values poses it.

  Pose only: rest geometry is never touched. Each of Location / Rotation /
  Scale has its own checkbox, so a node can move a bone without also pinning
  its rotation — an unchecked component is left exactly as the rig has it.

  The read happens **on selection**, not continuously: in a modifier stack the
  bone's position is an *output* of the graph, so a live read-back would race
  the pose the node writes. The refresh button re-reads on demand.

  The incoming rig arrives on its **Parent** input. Its Constraints input is
  pose-stack data, so it only reaches the armature in **Full Rig** mode.
- **Chain** — generates N connected bones from a start, direction, length and
  an optional per-segment curve. Inputs Parent and Tip Constraints.

### Transform

All four write the **pose**, never rest geometry — so they cannot change the
proportions of a rig they are layered onto, and a custom shape follows because
Blender draws widgets at the posed bone. A component the graph does not set
keeps whatever the rig already has.

- **Position** — set the selected bones' world position (Set Position).
- **Rotation** — set their world orientation.
- **Transform** — *offset* translation and rotation, so several Transform
  nodes stack, plus an optional absolute scale.
- **Snap** — put the selected bones on a mesh. Snap To picks what "on" means:
  - **Origin** — the target object's own origin
  - **Bounding Box** — centre of its bounds
  - **Median** — mean of its vertices
  - **Volume** — volume centroid, the centre of mass of a solid. Unlike the
    median this ignores how densely the mesh is subdivided.
  - **Surface** — closest point on the surface to the bone

  plus an offset applied after the snap.

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
| `baseline.py` | The rig's stored record of its unmodified state, kept on the armature object |
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
