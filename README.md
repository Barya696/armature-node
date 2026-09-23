# Armature Nodes — Procedural Armature Node System

A Blender addon that adds a node editor where a graph edits an armature, the
way Geometry Nodes edits a mesh. **Armature Input** hands the rig to
**Armature Output**, and every node in between is a modifier: it reads the
bone stream, changes the bones it selects, and passes the rest through.

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

Every modifier node has a **Bone** field. Leave it empty and the node affects
every bone on the wire; name one (or several, semicolon separated) and it
affects only those. When the graph has a rig to look at, the field is a
searchable dropdown of that rig's real bone names.

The graph stays the size of your edits, not the size of the rig.

## Node categories

### Armature I/O

- **Armature Input** — nothing in, Bone out. Reads every bone of the chosen
  armature: rest geometry, parenting, deform flags and existing widgets. It
  re-reads the live rig on every evaluation, which is what makes the stack a
  stack: the nodes downstream are the change, so the rig is the base state.
- **Armature Output** — Bone in, nothing out. Leaving *Armature* blank targets
  the Input's source, which is the normal case. Two modes:
  - **Modify** (default) — writes shapes and pose onto the existing rig; its
    bones, constraints and drivers are left alone. Safe on a Rigify rig.
  - **Full Rig** — the graph owns the armature and rebuilds its bones, so
    nodes can add and remove them.

### Marker

A marker is a world position with a stable key, drawn as a draggable empty in
the `MRKS_rig` collection. Placing things by dragging a handle beats typing
coordinates, which is the only reason markers exist.

- **Marker** — one handle. Inputs Parent and Constraints, outputs a Bone whose
  head sits at the handle and which runs along *Direction* for *Length*.
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

  Its Parent and Constraints inputs are rest/pose-stack data, so they only
  reach the armature when the Output is in **Full Rig** mode.
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
| `sockets.py` | Bone (the stream), Constraint, Vector sockets |
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

tree.links.new(src.outputs["Bone"], move.inputs["Bone"])
tree.links.new(move.outputs["Bone"], out.inputs["Bone"])

bpy.ops.armature_nodes.build()
```
