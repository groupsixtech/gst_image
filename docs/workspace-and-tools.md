# Workspace and drawing tools

The workspace has a central image canvas, the **Analysis** dock (normally left), and the
**Results and layers** dock (normally right). All drawing coordinates are tied to original source
pixels, even when the displayed overview is scaled.

## File and Analysis menus

| Command | What it does | Use it when |
| --- | --- | --- |
| **Open image…** (`Ctrl+O`) | Starts a new project in memory from an image. | Beginning a new analysis. |
| **Open project…** (`Ctrl+Shift+O`) | Reopens masks, recipes, ROIs, results, and strokes from a `.gstproj` directory. | Continuing or reviewing work. |
| **Save project…** (`Ctrl+S`) | Saves project metadata and masks; the source is referenced by path and checksum. | Maintaining reproducible work on the same machine. |
| **Save portable project as…** | Saves a project and copies the source image into it. | Moving work to another computer or person. |
| **Relink source as new revision…** | Associates a replacement source after confirmation and invalidates derived masks/runs. | The source pixels changed or moved. It deliberately does not silently reuse old results. |
| **Export…** (`Ctrl+E`) | Writes masks, tables, recipe/provenance JSON, fractions, native ROI source crops, per-layer segmentation images, and grouping overlays to a chosen folder. | Delivering data outside GST Image. |
| **Quit** | Closes the application. | After saving work. |
| **Analysis > Preview** (`Shift+A`) | Same as the Analysis panel’s Preview button. | Quickly retesting after a small adjustment. |

**Edit > Undo/Redo** reverses or reapplies eligible manual paint edits. **Decrease brush radius**
(`[`), **Increase brush radius** (`]`) adjust a brush when a brush-like tool is active.

## Canvas navigation

- Select **Pan** to drag the view normally. The mouse wheel zooms around the cursor.
- Hold the middle mouse button and drag to pan temporarily from any tool.
- The status bar reports source-image `x, y` coordinates under the pointer.
- When a brush tool is active, the pink circle is its source-pixel footprint at the current zoom.

## Toolbar tools

| Tool | Canvas action | Result and advice |
| --- | --- | --- |
| **Pan** | Drag. | Moves the view only. |
| **Select** | Click an instance on the active particle layer. | Selects particles for inspection, **Merge selected**, or **Split**. Select at least two for merge and exactly one for split. |
| **Calibrate** | Drag along a known scale bar. | Prompts for known length and unit. Draw from end to end, not along its surrounding border. |
| **Measure** | Drag a line. | Prompts for a name and records pixel length plus physical length when calibrated. |
| **Eyedropper** | Click an image location. | Sets the current target selected in **Eyedropper sets**: manual lower/upper threshold, segmentation-class colour, or region-class colour. It samples a native-resolution 5×5 neighbourhood, so click a clean representative patch. |
| **Include box** | Drag a rectangle. | Adds a rectangular included ROI. Multiple include/analysis ROIs form a union before exclusions are subtracted. |
| **Analysis box** | Drag a rectangle. | Adds a rectangular ROI and saves the current particle recipe with that box. Selecting it later restores that recipe; useful when different areas require different settings. |
| **Exclude box** | Drag a rectangle. | Removes that area from every analysis domain: use it for scale bars, text, black seams, and artifacts. |
| **Polygon** | Click vertices; double-click after at least three vertices. | Adds an included polygon. Use for irregular specimen boundaries. |
| **Class seed** | Paint example strokes. | Adds human-labelled training pixels for the selected assisted-region class. It is training input, not a result-mask edit. See [region classification](region-classification.md). |
| **Seed eraser** | Paint over seed strokes. | Removes training pixels for the next classifier training. |
| **Mask brush** (`O`) | Paint an active saved layer. | For particles, adds foreground; for a region layer, paints the selected class. Select the result layer first. |
| **Mask eraser** (`P`) | Paint an active saved layer. | Clears particle foreground or region labels. |
| **Split** | Draw across one selected particle. | Cuts a selected instance along the line. Use a narrow line across the neck, then inspect both pieces. |
| **Merge selected** | Click toolbar action after Select-ing instances. | Combines two or more selected particle labels. Use only for parts known to be one object. |

## Results and layers panel

### Summary

Shows the last preview or run summary. For a completed particle run, key fields include particle
count, analysis-domain pixels/area, segmented pixels/area, area fraction, estimated volume
fraction percentage, size statistics, and number density. By default, border-touching particles
remain in the mask and area fraction but are omitted from size statistics because only part of
them is visible.

### Particles

Each row is one labelled instance. **Size** defaults to equivalent radius and changes to the
active grouping metric when a grouping is displayed. Circularity is 1 for an ideal circle and
falls toward 0 for elongated/jagged shapes; solidity compares an object’s area to its convex
hull; **Border** identifies objects touching the analysis-domain boundary. This table is an
inspection aid; segmentation-area filters happen earlier in the recipe.

### Layers

Click a layer to make it active. Tick/untick its checkbox to show/hide it, and set **Selected
layer opacity** to make an overlay more or less transparent. Instance layers are particle results;
multiclass layers are region classifications; domain layers record the denominator used for a
matching result and are normally hidden. **Delete selected** removes selected layers and dependent
records. **Copy ROI mask image** places a native-resolution black-and-white crop of the active
result on the clipboard.

### ROIs

Lists each ROI and its kind. Clicking an Include or Analysis ROI selects it, turns on
selected-ROI analysis, restores its saved recipe when present, and selects its latest matching
particle result (or its region-classification result when no particle result exists). Region
classification requires exactly one selected Analysis box. **Delete selected** removes the ROI
and layers/results derived from it, so save or export first if those data are needed.

### Measurements

Lists calibration and recorded line measurements. **Delete selected** removes selected line
measurements; calibration is represented separately and should be recreated if a wrong scale was
used.

### Particle groups

This tab is documented in [review, correction, grouping, and export]
(review-correction-grouping-export.md). It works only with an active instance layer and never
changes the underlying segmentation. It can save a native-resolution grouping-overlay PNG for
the Analysis box and save its colour graphs as individual/combined PNGs plus a Matplotlib Figure
pickle.
