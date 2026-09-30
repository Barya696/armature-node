# Armature Nodes — Procedural Armature Node System

A Blender addon that adds a node editor where a graph edits an armature, the
way Geometry Nodes edits a mesh. **Armature Input** puts the whole rig on the
wire, **Armature Output** writes it back, and every node in between is a
modifier: one value in, one value out, changing only what it selects.

## Install

Requires Blender 4.2 or later; tested on 5.1 and 5.2.

It is a Blender **extension** (`blender_manifest.toml`):

1. Build the package from this folder:
   `blender --command extension build --source-dir . --output-dir dist`
2. Drag `dist/armature_nodes-<version>.zip` into Blender, or use
   `Edit > Preferences > Get Extensions > Install from Disk...`.

The folder also still works as a legacy add-on: zip the `armature_nodes/`
folder itself and use `Preferences > Add-ons > Install from Disk...`.

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

Several wires can go into the Output — one branch per Transform node, say,
each moving its own bone. They are **chained, in the order they were plugged
in**: each branch runs on top of the ones before it, from the node where it
split off from them. So nodes side by side do exactly what they would one
after another: every branch's changes arrive, offsets on the same bone add
up, a value two branches both set is taken from the wire plugged in last,
and a node the branches share runs once.

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
| `an_rig_record` | the complete rig as JSON: bones, hierarchy, deform flags, collections, colours, display, pose settings, switches (number and on/off custom properties), full constraint properties |
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

A marker is a world position with a stable key, drawn as a glowing light and
backed by an empty in the `MRKS_rig` collection. Placing things by dragging a
handle beats typing coordinates, which is the only reason markers exist.

**Grab the glow — in any mode, Pose mode included.** The glow is a viewport
gizmo, like the ones on a light or a camera, so it does not need the empty to
be clickable (Blender will not let you click another object while the
armature is in Pose mode). It lights up with the marker's name beside it
when the mouse is over it.

**Handles are sized in the scene, like the character they sit on**: zoom in
and they grow, zoom out and they shrink — so a zoomed-out figure is not
buried under glows, and a zoomed-in hand gets handles you can tell apart. A
handle is a fixed share of its figure (the Skeleton's landmarks, or a Marker
node's rig at rest), times the node's **Size**. Zoomed far out it stops
shrinking at a size you can still see and grab.

- **Drag the glow** to move the marker. **Ctrl** drops it onto the surface
  under the cursor, **Shift** moves it finely, **X / Y / Z** lock the move to
  that axis. **Esc** or right-click puts it back.
- **Drag the ring** — there when the marker supplies a rotation — to turn it
  about the view, or with X / Y / Z about that axis. Ctrl: 5° steps.
- **Drag the square** on the ring — there when it supplies a scale — to scale
  it. X / Y / Z: that axis only. Ctrl: steps of 0.1.
- A marker whose position is only a readout of the bone turns (or scales)
  when you drag its glow instead.
- Every drag is one undo step.

The drag moves the marker's empty, exactly as grabbing the empty would, so
locks, mirroring, rigid groups and the live link to the bone all apply.

Each marker node has a **Size** slider for its handles, and a Marker node has
a **colour** swatch for its glow (MediaPipe landmarks keep their side
colours).

**Parent and child, like bones.** A Marker node has a **Parent** input. Wire
another marker's output into it — a Marker node, or a Skeleton landmark — and
this marker becomes its child:

- Its values become **relative to the parent**, like a child bone's Location
  and Rotation in the N-panel; the node says *Relative to its parent*.
- It **moves, turns and scales with the parent**: move the parent and it
  follows, turn the parent and it swings around it. Drag the child on its own
  and only the child moves.
- A **line** is drawn from the parent to the child, in the child's colour.
  Chain as many as you like: root → middle → tip.
- Connecting or removing a parent **keeps the marker where it is** (like
  Ctrl+P / Alt+P with Keep Transform).
- The nodes a child feeds still get its **world** transform, so a child
  marker drives its bone to where you see it.

With the markers live on a bone chain, grabbing the parent bone carries the
child bone — and the child marker comes along exactly once, not also by the
carry: marker updates in a live pass land together, parents first.

**Lines between markers, as many as you like.** A line joins two Marker
nodes, and a marker can be joined to any number of others. A line is only a
line: nothing follows anything (for a marker that moves with another, use
Parent).

- **In the node editor** each end of a line is a port on the node's border,
  facing the other node, and it slides round the node as either one moves —
  the way DaVinci Resolve's Fusion page draws its connections. On a node's
  sides the ports stay below Blender's own sockets, so they never cover one.
- **To join two markers**, drag from the small ring at the bottom of a
  Marker node and drop on another Marker node. While you drag, the wire snaps
  to the node it would join and outlines it.
- **Drag a line's end** onto another Marker node to move it there, or onto
  empty space to remove it. A click leaves it alone. Every drag is one undo
  step.
- **In the viewport** the line runs between the two handles, shading from one
  marker's colour to the other's, whenever both are shown.
- **Node menu / right-click:** *Join Markers* joins the active Marker node to
  every other selected one; *Remove Marker Lines* removes the lines among the
  selected. The sidebar (N › Node) lists a Marker node's lines, each with a
  remove button.
- Lines survive renaming. A deleted marker takes its lines with it. Shift+D
  copies the lines among the markers it copies. Making a group or ungrouping
  takes a line along when both its markers go; a line to a marker left
  outside a group is removed, because a line joins two markers in the same
  tree.

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
  - **Wiring it in never moves the bone.** The marker first takes the
    bone's live values — or, for an Offset or a Local-space Transform field,
    the value the field held.
  - **Grab, rotate or scale the bone** in Pose mode and the marker and its
    handle follow.
  - **Drag, turn or scale the handle, or type a value**, and the bone follows.
  - **Unplug or delete it** and the field takes its value, so the bone stays.
  - Each value is labelled by the input it feeds (*Location*, *Rotation*,
    *Scale*, *Offset*…). One that drives nothing is a greyed readout of the bone, and
    that part of the handle is locked.

  The handle always sits on the bone. For a value measured from rest (an
  Offset, a Local-space Transform field) it is drawn where the value puts the
  bone — rest plus the value — rather than at the raw offset, which would
  leave it near the world origin.
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

- **Wrap Markers** — fits the tree's markers onto a character mesh, the
  way the wrap add-on fits a template onto a scan. Two buttons and a switch,
  on the node and in the 3D Viewport sidebar's **Wrap** tab:
  - **Fit to Mesh** — select the rig and the character, in either order,
    and press it. The markers it knows by name (the Human Skeleton's, below:
    Pelvis, Chest, Head, Clavicles, Hands, Feet) are paired with their
    places automatically, found from human proportions and, for the hands,
    the tips of the arms; then the skeleton is snapped onto the pairs, drawn
    into the middle of the limbs and stuck there. The line above the buttons
    says which character it fits to, and when it has.
  - **Pick Pairs** — for a marker the automatic pairs miss or place wrong.
    Press it and it stays pressed in, the picker on: over the viewport the
    cursor turns into an eyedropper. Click a marker's glow — the cursor turns
    into a crosshair — then where it goes on the character, then the next. A
    joint goes *into* the mesh, halfway through it under the cursor. With
    the mirror toggle to its left on (it is by default), its partner (Hand.R
    for Hand.L) gets the mirror image across the character's middle — both
    places show under the cursor before you click. Turn it off for a
    character that isn't symmetric: Fit then leaves each side to itself too.
    While it runs, **✕** beside it (or right-click) unselects the picked
    marker and **⟲** (or **Ctrl Z**) takes back the last pair, its marker
    picked again to put it right; **X** forgets a marker's pair. Press the button again, or **Esc**, to stop; the whole
    session is one undo step. Fit keeps what you picked and redoes the rest
    every time, so it follows a character moved since; selecting another
    character forgets the last one's picks.
  - **Original | Wrapped** — at the bottom: the skeleton as it was before
    the fit, or fitted, to compare the two. Each keeps what you change while
    it shows — a marker touched up after the fit is still touched up when
    you come back to Wrapped — and Fit fits the skeleton as Original shows
    it.

  How the skeleton moves: its bones are the lines the viewport draws between
  the markers (a marker and its parent, joined markers, MediaPipe's bones).
  Every bone keeps its length, a marker where three or more bones meet (a
  chest, a pelvis) turns as one solid piece, and the joints between bend:
  hands paired into an A-pose swing the arms down at the shoulders instead
  of dragging the chest. Pole targets (the Human Skeleton's Elbows and
  Knees) are carried, never drawn into the mesh. A marker moves the way a
  drag moves it — a child against its parent, a face or finger landmark with
  its anchor.

  The fit streams into the markers, so the skeleton glides onto the
  character and the rig follows live; **Esc** stops it and puts them back,
  and each fit is one undo step. In the sidebar, a rig with no markers yet
  gets **Add Human Skeleton**, and one with markers but no Wrap Markers node
  gets **Wrap These Markers**. The single steps (Snap, Attract, Stick),
  Symmetrize and each marker's role (Inside, Surface, Free, Fixed) are still
  there for scripts: `armature_nodes.wrap_run`, `wrap_symmetrize`,
  `SkeletonMarker.wrap_role`; the switch is the node's `preview`.
- **Human Skeleton** (Shift+A › Group) — a ready-made group: thirteen
  markers — Pelvis, Chest, Head, Clavicles, Elbows, Hands, Knees, Feet —
  each driving the Rigify control that does that job through a Transform
  node (torso, chest, head, shoulder, the IK hands and feet, the elbow and
  knee pole targets), parented like the bones, the thighs drawn as lines. A
  Rigify Switch turns the poles on and a Wrap Markers node comes with it.
  The group is made the first time it is added. Wired into a Rigify rig's
  tree, every marker takes its control's place, so nothing moves until you
  fit it to a character: select the rig and the character, **Wrap** tab,
  **Fit to Mesh**.

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
- **Transform** — the bone's **Location, Rotation and Scale**, live. Pick a
  bone and the fields fill with where it is; nothing is applied yet. A value
  you have not set is a readout that follows the bone, so a fresh node
  changes nothing. **Type a value**, or turn on its **Set** toggle, and the
  node sets that part of the bone — turning it on starts from where the bone
  is, so it never jumps. Parts left unset keep whatever the bone has. A later
  node that sets the same part wins. **Space**:
  - **World** — the bone's world location and rotation.
  - **Local** — its own channels, the N-panel values, which follow the parent
    the way hand-posing does. Several bones in one node each get the same
    channel values, as typing them into each bone's N-panel would.

  Scale is the bone's own in both (1 at rest), never the armature object's.

  The **Transform** input is the same thing on one wire, in world space.
  Wire a Marker into it and the marker first takes the bone's live Location,
  Rotation and Scale — nothing moves — and from then on the handle and the
  bone follow each other. While it is wired the three fields are hidden and
  Space does not apply; unplug it and the fields take its values, set, so
  the bone stays where it is. A source that carries no rotation — a Skeleton
  landmark with rotation off — moves the bone without turning it.

  A Transform node saved when its fields were offsets from rest (*Translation*)
  is converted on first use: each part it applied is set, and re-read from
  the bone, which is where the offset put it.
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
- **Node to rig**: type a value and the bone moves. On Position, Rotation and
  Transform, typing into the field takes the bone over (turns *Set* on) — a field that
  looked like an input but only displayed was the old complaint.
- While a node is not driving a value (*Set* off, nothing wired, zero
  offset), that field is a plain readout of where the bone is.

The node tells its own writes apart from yours with a snapshot taken after
every build: anything that differs from it was you, and only that difference
is folded in. That is what keeps it from chasing its own output — which on a
constrained bone would oscillate for ever. Relative values (Offset, Local)
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
- **A later absolute value wins.** Set Position after an Offset replaces it,
  as in Geometry Nodes.
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

### Rigify

- **Rigify Switch** — sets the rig's own switches from the graph: Rigify's
  IK / FK, Pole, IK Parent, Pole Parent, IK Stretch and FK Limb Follow (kept
  on a limb's `*_parent` bone: `upper_arm_parent.L`, `thigh_parent.L`), Neck
  Follow, Head Follow and Torso Parent (on `torso`) — and any other number or
  on/off custom property a bone has.
  - Pick the bone with the ▾ menu, which lists only bones that have
    switches, or leave **Bone** empty to set a switch on **every** bone that
    has it: one node turns the poles on for all four limbs.
  - Tick a switch to set it; unticked, it shows what the rig has. Ticking
    starts from the rig's value, so nothing jumps. A parent switch is a
    dropdown of Rigify's own names (None, Root, Torso, Hips…).
  - Like every node it works over the record: **deleting it puts back** the
    values the rig was bound with. A switch changed in Rigify's own panel is
    taken into the node, so the next build keeps it.
  - A rig bound before switches were recorded shows **Record Switches**,
    which adds them to its record and changes nothing else.

### Node groups

The same principle as Blender's own node groups. A group is a node tree of its
own (a data-block in *Blender File > Node Groups*), used from other trees
through a **group node**:

- **Ctrl+G** puts the selected nodes into a new group and opens it. Every wire
  that crossed the edge of the selection becomes one of the group's inputs or
  outputs — one per source, named after the socket it feeds — and the group
  node takes the nodes' place, wired as they were. The rig does not change.
- **Tab** goes into the selected group node, and back out; **Ctrl+Tab** only
  goes out. **Ctrl+Alt+G** puts a group node's nodes back in its place.
- Inside, **Group Input** puts the group's inputs on wires and **Group
  Output** takes its outputs. Drag a wire into their empty socket to add one;
  rename, reorder or remove them in the sidebar's **Group** tab (*Group
  Sockets*), as in any Blender node group.
- **Shift+A > Group** adds Group Input / Output (inside a group), the
  ready-made **Human Skeleton** (see *Marker*), and any existing group —
  except one that would end up inside itself.
- One group can be used by any number of group nodes, on any rig. **Edit it
  once and every one of them changes**, and every rig using it rebuilds.
- Values cross the edge the way wires do: a marker outside wired into a group
  node drives the node inside, and a value typed on the group node's socket
  is used when nothing is wired in. Groups can hold groups.

**Markers go into groups too.** A marker inside a group shows its handle in
the viewport when the group node using it feeds an Armature Output, and you
drag it as usual. Like any value inside a Blender node group, it is shared:
every group node running the group uses the same marker.

**A group used once is live**, exactly like the rig's own tree: a marker
wired in takes the bone's place, grabbing the bone moves the marker, picking
a bone fills the fields. A group used by several group nodes cannot know
which bone to follow — so there its markers and values still drive the rig,
but nothing follows a grab (the Bone field still lists a rig's bones). A
marker left outside and wired into a group node drives the nodes inside, but
is not live with them.

## Execution model

0. **Baseline** — the Armature Input emits the rig's recorded, unmodified
   state (captured by **Bind Rig**, never implicitly).
1. **Evaluate** — walk back from the Armature Output, each node once
   (memoized), into a list of `BoneDef`. The bones are shared down the stream
   and a node copies only the ones it changes; several wires into the Output
   are chained (see *One value on the wire*).
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
| `core.py` | `BoneDef` / `ConstraintDef` / `ShapeDef`, the evaluator (memoization, groups, chaining several wires), copy-on-write, link index |
| `model/` | Pure data: types, JSON schema, v1→v2 migration, diff. Imports no `bpy`. |
| `store/` | The `an_rig_*` properties, and the build lock. The only place they are touched. |
| `capture/` | Live armature → `RigRecord`. Never writes. Called only from Bind and Capture. |
| `apply/` | `RigRecord` → live armature, and the Full Rig writers (`full_rig.py`). The only writer. |
| `bridge.py` | The seam between the graph's `BoneDef` and the record's `RigRecord` |
| `livelink.py` | Telling the graph's own pose writes from the user's, for the two-way live link |
| `sockets.py` | Rig (the whole armature, the stream), Constraint, Vector / Rotation / Scale / Transform sockets |
| `tree.py` | `ArmatureNodeTree` data-block, dirty tracking, debounced live update, graph version |
| `nodes/` | The node types, one module per Shift+A category (`armature_io`, `bone`, `marker`, `transform`, `shape`, `constraint`, `rigify`, and `wrap`, the Wrap Markers node, listed under Marker); `base.py` and `marker_base.py` hold what they share, `__init__.py` registers them |
| `primary_rig.py` | Marker handles and locks, viewport overlay, MediaPipe preset table, marker operators |
| `wrap_solver.py` | The Wrap Markers node's maths: Snap, Attract and Stick over a marker skeleton. numpy, no `bpy` |
| `human_skeleton.py` | The Human Skeleton group, made on first use, and where its markers go on a character (Auto) |
| `handles.py` | The grabbable marker gizmo: hit-testing, move / turn / scale drags, rings and name label |
| `marker_links.py` | Lines between Marker nodes: the joins, Fusion-style ports, their drawing and drag gizmo |
| `groups.py` | Node groups: the group node, Make Group / Ungroup / Tab, group sockets kept in step, Shift+A > Group |
| `widgets.py` | `WGTS_rig` widget library and Rigify-style preset generation |
| `build.py` | A build: evaluate, then Modify (restore → apply → record) or Full Rig |
| `decompile.py` | Reverse: the two-node stack that targets a rig |
| `operators.py` | Build, Convert to Armature Nodes, Read From Rig |
| `ops/` | Bind Rig, Capture, Record Switches, Restore Original, Forget |
| `ui.py` | Shift+A categories, header buttons, N-panel sidebar |
| `sync.py` | Editor follows the active armature; the rig and marker handles read back when they move |

## Performance

What runs, and when — the numbers are from a 706-bone Rigify rig with 30 nodes:

- **A node edit** rebuilds after an 80 ms debounce, in about 17 ms. The rig's
  record is parsed once and cached against its stored text; nodes share the
  rig's bones and copy only the ones they change; the diff skips every bone
  that is the very same object as the record's.
- **Posing a bone or dragging a marker** reads the rig back into the live
  nodes, from `depsgraph_update_post` — only when the update moved a bound
  rig or a marker handle. Anything else (a node dragged in the editor, a
  material changed) costs one pass over the update list. A dragged handle
  rebuilds on the next frame, not after the debounce, so the rig keeps up
  with the mouse: about 20 ms a frame for a control, 45 ms for the root.
- **Redrawing** the viewport or node editor reads cached results: which
  markers are displayed, the node links, the record. They are rebuilt when a
  graph changes (`tree.graph_version`), and dropped on undo and file load.
- **Nothing polls** except what Blender cannot report — an editor switched to
  Armature Nodes, an armature changing mode — twice a second, in well under a
  millisecond.

## Development

Tests, from this folder:

```
blender -b --factory-startup -P tests/blender/run.py
python tests/run_pure.py
```

The Blender suite takes one module by name: `... run.py -- test_groups`.
The pure suite needs numpy for the wrap solver's tests; Blender ships it.
A new node goes in the `nodes/` module of its Shift+A category, and into
`classes` in `nodes/__init__.py`.
Warnings go to the system console through the `armature_nodes` logger.
Disabling the add-on drops all of its modules, so enabling it again (or
Reload Scripts) runs the code as it is on disk.

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
