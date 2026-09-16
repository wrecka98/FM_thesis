from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from pathlib import Path

OUT = Path("artifacts/training_hyperparameters.docx")
BLUE = "2E74B5"
DARK = "1F4D78"
INK = "24364B"
MUTED = "667085"
HEADER = "E8EEF5"
LIGHT = "F4F6F9"
WHITE = "FFFFFF"


def shade(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def borders(table, color="C9D3DF", size="4"):
    tbl_pr = table._tbl.tblPr
    old = tbl_pr.find(qn("w:tblBorders"))
    if old is not None:
        tbl_pr.remove(old)
    el = OxmlElement("w:tblBorders")
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single")
        e.set(qn("w:sz"), size)
        e.set(qn("w:color"), color)
        el.append(e)
    tbl_pr.append(el)


def cell_margins(cell, top=80, start=120, bottom=80, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    mar = tc_pr.first_child_found_in("w:tcMar")
    if mar is None:
        mar = OxmlElement("w:tcMar")
        tc_pr.append(mar)
    for tag, val in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = mar.find(qn(f"w:{tag}"))
        if node is None:
            node = OxmlElement(f"w:{tag}")
            mar.append(node)
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")


def set_fixed_table(table, widths):
    table.autofit = False
    table.alignment = WD_TABLE_ALIGNMENT.LEFT
    tbl_pr = table._tbl.tblPr
    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")
    tbl_w = tbl_pr.find(qn("w:tblW"))
    tbl_w.set(qn("w:w"), str(sum(widths)))
    tbl_w.set(qn("w:type"), "dxa")
    ind = OxmlElement("w:tblInd")
    ind.set(qn("w:w"), "120")
    ind.set(qn("w:type"), "dxa")
    tbl_pr.append(ind)
    grid = table._tbl.tblGrid
    for child in list(grid):
        grid.remove(child)
    for width in widths:
        col = OxmlElement("w:gridCol")
        col.set(qn("w:w"), str(width))
        grid.append(col)
    for row in table.rows:
        for i, cell in enumerate(row.cells):
            cell.width = Inches(widths[i] / 1440)
            tc_w = cell._tc.get_or_add_tcPr().find(qn("w:tcW"))
            tc_w.set(qn("w:w"), str(widths[i]))
            tc_w.set(qn("w:type"), "dxa")
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cell_margins(cell)
    # Mark the first row as repeating/header metadata. This also gives the
    # one-row callout tables a discoverable lead row for accessibility tools.
    tr_pr = table.rows[0]._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def add_table(doc, rows, widths=(2700, 6660), headers=("Parameter", "Value / implementation")):
    table = doc.add_table(rows=1, cols=2)
    table.style = "Table Grid"
    h = table.rows[0].cells
    for i, value in enumerate(headers):
        p = h[i].paragraphs[0]
        p.paragraph_format.space_after = Pt(0)
        r = p.add_run(value)
        r.bold = True
        r.font.color.rgb = RGBColor.from_string(INK)
        shade(h[i], HEADER)
    for idx, (key, value) in enumerate(rows):
        cells = table.add_row().cells
        cells[0].paragraphs[0].add_run(str(key)).bold = True
        cells[1].paragraphs[0].add_run(str(value))
        if idx % 2:
            shade(cells[0], "FAFBFC")
            shade(cells[1], "FAFBFC")
        for cell in cells:
            for p in cell.paragraphs:
                p.paragraph_format.space_after = Pt(0)
                p.paragraph_format.line_spacing = 1.05
                for r in p.runs:
                    r.font.size = Pt(9.2)
    set_fixed_table(table, widths)
    borders(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(0)
    return table


def add_note(doc, label, text):
    table = doc.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    shade(cell, LIGHT)
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(0)
    r = p.add_run(label + " ")
    r.bold = True
    r.font.color.rgb = RGBColor.from_string(DARK)
    p.add_run(text)
    set_fixed_table(table, (9360,))
    borders(table, color="D8E0E8", size="3")
    doc.add_paragraph().paragraph_format.space_after = Pt(0)


def source(doc, text):
    p = doc.add_paragraph(style="Source")
    p.add_run(text)


doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.5), Inches(11)
sec.top_margin = sec.bottom_margin = sec.left_margin = sec.right_margin = Inches(1)
sec.header_distance = sec.footer_distance = Inches(0.492)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11)
normal.font.color.rgb = RGBColor.from_string("202A35")
normal.paragraph_format.space_after = Pt(6)
normal.paragraph_format.line_spacing = 1.25
for name, size, color, before, after in (
    ("Title", 25, INK, 0, 6),
    ("Subtitle", 13, MUTED, 0, 14),
    ("Heading 1", 16, BLUE, 18, 10),
    ("Heading 2", 13, BLUE, 14, 7),
    ("Heading 3", 12, DARK, 10, 5),
):
    s = styles[name]
    s.font.name = "Calibri"
    s.font.size = Pt(size)
    s.font.color.rgb = RGBColor.from_string(color)
    s.font.bold = name != "Subtitle"
    s.paragraph_format.space_before = Pt(before)
    s.paragraph_format.space_after = Pt(after)
    s.paragraph_format.keep_with_next = True
styles.add_style("Source", 1)
styles["Source"].font.name = "Calibri"
styles["Source"].font.size = Pt(8.5)
styles["Source"].font.color.rgb = RGBColor.from_string(MUTED)
styles["Source"].paragraph_format.space_before = Pt(4)
styles["Source"].paragraph_format.space_after = Pt(4)

header = sec.header.paragraphs[0]
header.text = "TRAINING PARAMETER RECORD  |  FM_thesis"
header.style = styles["Source"]
header.paragraph_format.space_after = Pt(0)
footer = sec.footer.paragraphs[0]
footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
footer.add_run("Reproducibility appendix  •  generated 14 September 2026")
footer.style = styles["Source"]

doc.add_paragraph().paragraph_format.space_after = Pt(28)
p = doc.add_paragraph("TRAINING PARAMETER RECORD", style="Title")
p.add_run("\n")
doc.add_paragraph("nnU-Net training, VersaMammo segmentation-head adaptation, and MedSAM LoRA fine-tuning", style="Subtitle")
p = doc.add_paragraph()
p.paragraph_format.space_after = Pt(2)
p.add_run("Scope: ").bold = True
p.add_run("parameters recoverable from the checked-in experiment plans, runtime debug records, training scripts, notebooks, and result metadata.")
p = doc.add_paragraph()
p.add_run("Repository: ").bold = True
p.add_run("FM_thesis")
p = doc.add_paragraph()
p.add_run("Evidence date: ").bold = True
p.add_run("14 September 2026")
add_note(doc, "Reading convention.", "Recorded values come from a run artifact or executable experiment notebook. Code-default values are the current defaults in the training implementation and are not independently preserved in the corresponding result artifact. These two evidence levels are kept distinct throughout.")

doc.add_heading("1. At-a-glance configuration", level=1)
add_table(doc, [
    ("nnU-Net", "2-D PlainConvUNet; batch 2; 1,000 planned epochs; SGD, lr 0.01, momentum 0.99, Nesterov; weight decay 3×10⁻⁵; Dice + cross-entropy with deep supervision."),
    ("VersaMammo head", "UNetEfficientNetB5 initialized from VersaMammo (Enb5); encoder/backbone frozen; adaptation/decoder head trainable; 100 epochs; AdamW; lr 10⁻³; BCE + Dice; flip augmentation."),
    ("MedSAM LoRA", "MedSAM ViT-B; LoRA on qkv modules with rank 8, alpha 8, dropout 0.1, no bias; 50 epochs; AdamW, lr 10⁻⁴, weight decay 0.01; Dice + BCE-with-logits."),
])

doc.add_page_break()
doc.add_heading("2. nnU-Net", level=1)
doc.add_paragraph("The experiment uses nnUNetTrainer with the nnUNetPlans 2-D configuration for Dataset001_Mammography. Runtime debug files corroborate the trainer and plan values below.")
doc.add_heading("2.1 Data, configuration, and sampling", level=2)
add_table(doc, [
    ("Dataset", "Dataset001_Mammography; 102 training cases; one grayscale input channel; labels: background = 0, lesion = 1; PNG input."),
    ("Cross-validation", "Folds 0–4 are defined in splits_final.json. Runtime debug artifacts are present for folds 0–3 at the evidence date."),
    ("Configuration", "2-D; data identifier nnUNetPlans_2d; DefaultPreprocessor; NaturalImage2DIO."),
    ("Patch size", "1,792 × 896 pixels."),
    ("Batch size", "2."),
    ("Target spacing", "[1.0, 1.0]."),
    ("Normalization", "Z-score normalization; use_mask_for_norm = false."),
    ("Foreground oversampling", "0.33; probabilistic oversampling disabled."),
    ("Iterations", "250 training iterations/epoch; 50 validation iterations/epoch."),
    ("Training workers", "12 augmentation processes; 6 validation processes (runtime fold-0 debug record)."),
])
source(doc, "Sources: nnUNet/.../plans.json; nnUNet/.../fold_0/debug.json; nnUNet_preprocessed/.../splits_final.json.")

doc.add_heading("2.2 Network architecture", level=2)
add_table(doc, [
    ("Network", "dynamic_network_architectures PlainConvUNet; 2-D convolutions; 1 input channel; deep supervision enabled."),
    ("Stages", "9 encoder stages; features per stage: [32, 64, 128, 256, 512, 512, 512, 512, 512]."),
    ("Kernels", "3 × 3 at every stage."),
    ("Strides", "[(1,1), seven × (2,2), (2,1)]."),
    ("Convolutions/stage", "Encoder: 2 at each of 9 stages. Decoder: 2 at each of 8 stages."),
    ("Normalization", "InstanceNorm2d, epsilon 10⁻⁵, affine = true."),
    ("Activation", "LeakyReLU, inplace = true. Negative slope is not explicitly set in the plan and therefore follows the library default."),
    ("Bias / dropout", "Convolution bias enabled; dropout disabled (dropout_op = null)."),
    ("Compilation", "torch.compile enabled by default and runtime network recorded as OptimizedModule."),
])
source(doc, "Sources: nnUNet_results/.../plans.json and fold_0/debug.json; nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py.")

doc.add_page_break()
doc.add_heading("2.3 Optimization, objective, and checkpointing", level=2)
add_table(doc, [
    ("Planned duration", "1,000 epochs (250,000 training iterations if completed without interruption)."),
    ("Optimizer", "SGD; initial learning rate 0.01; momentum 0.99; Nesterov enabled; dampening 0; weight decay 3×10⁻⁵."),
    ("Learning-rate schedule", "Polynomial decay by epoch: lr = 0.01 × (1 − epoch/1000)^0.9. No warm-up is implemented."),
    ("Loss", "DC_and_CE_loss: memory-efficient soft Dice + robust cross-entropy; batch Dice enabled."),
    ("Deep supervision", "Enabled. Output weights decrease by 1/2 per lower-resolution output and are normalized to sum to 1; the lowest-resolution output receives zero weight in the trainer implementation."),
    ("Mixed precision", "CUDA GradScaler present in runtime debug record."),
    ("Checkpoint cadence", "Periodic checkpoint every 50 epochs; latest, best, and final checkpoints follow nnU-Net trainer behavior; checkpointing enabled."),
    ("Device recorded", "cuda:0 on NVIDIA GB10; PyTorch 2.11.0+cu130; cuDNN 91900 (fold-0 runtime record)."),
])
source(doc, "Sources: fold_0/debug.json; nnUNetTrainer.py; polylr.py.")

doc.add_heading("2.4 Training augmentation", level=2)
add_table(doc, [
    ("Spatial", "No elastic deformation; rotation probability 0.2, angle ±15°; scaling probability 0.2, factor 0.7–1.4; synchronized scaling across axes; mirroring on axes 0 and 1."),
    ("Gaussian noise", "Probability 0.1; variance 0–0.1; applied per channel with probability 1."),
    ("Gaussian blur", "Probability 0.2; sigma 0.5–1.0; per-channel probability 0.5."),
    ("Brightness", "Multiplicative brightness probability 0.15; multiplier 0.75–1.25."),
    ("Contrast", "Probability 0.15; range 0.75–1.25; preserve intensity range."),
    ("Low resolution", "Probability 0.25; scale 0.5–1.0; per-channel probability 0.5."),
    ("Gamma", "Inverted-image gamma: p = 0.1; regular gamma: p = 0.3; gamma 0.7–1.5; retain statistics."),
    ("Crop / padding", "random_crop = false; image padding = zeros; segmentation padding label = −1 then remapped to 0."),
])
source(doc, "Source: serialized training transform pipeline in fold_0/debug.json.")

doc.add_page_break()
doc.add_heading("3. VersaMammo adaptation head", level=1)
doc.add_paragraph("This section describes the segmentation adaptation implemented in versamammo_train_seg_v2.py and the ZGT head-training record. The result summary preserves the mode, duration, augmentation status, case count, and checkpoint provenance; optimizer-related values below are current code defaults because those values are not serialized in the summary.")
doc.add_heading("3.1 Model and trainable scope", level=2)
add_table(doc, [
    ("Model", "UNetEfficientNetB5 segmentation model."),
    ("Initialization", "Pretrained VersaMammo (Enb5) checkpoint; recorded ZGT source: VersaMammo/downstream/Sotas/VersaMammo (Enb5).pth."),
    ("Fine-tuning mode", "head (recorded). Parameters whose names begin with backbone. or encoder are frozen; all remaining parameters are trainable."),
    ("Output", "Binary foreground probability used directly by binary cross-entropy; evaluation threshold 0.5."),
    ("Training cases", "102 samples in the recorded ZGT run."),
    ("Device", "cuda:0 (recorded)."),
])
source(doc, "Sources: versamammo_train_seg_v2.py; SEG_ZGT_versamammo_segmentation/training_summary.json.")

doc.add_heading("3.2 Optimization and stopping", level=2)
add_table(doc, [
    ("Epochs", "100 completed in the recorded ZGT run; 1,300 iterations completed."),
    ("Optimizer", "AdamW (code default/implementation)."),
    ("Learning rate", "1×10⁻³ (code default; not serialized in the ZGT training summary)."),
    ("Weight decay", "0.0 (code default; not serialized in the ZGT training summary)."),
    ("Scheduler / warm-up", "None implemented."),
    ("Training batch", "8 (code default; not serialized in the ZGT training summary)."),
    ("Evaluation batch", "1 (code default)."),
    ("Iteration ceiling", "10,000,000 (code default; non-binding for the recorded 1,300-iteration run)."),
    ("Validation", "Auto: use Eval split if present; validate every 500 iterations. Best model selected by Dice. Early stop after 20 consecutive non-improving validations. If Eval is absent, save final state and disable early stopping."),
    ("Logging", "Every 25 iterations."),
    ("Random seed", "0 (code default)."),
])

doc.add_heading("3.3 Input processing, augmentation, and loss", level=2)
add_table(doc, [
    ("Input size", "512 × 512."),
    ("Intensity normalization", "Default percentile scaling: finite, nonzero pixels clipped to the 1st–99th percentiles and scaled to [0,1]; then channel-wise Normalize(mean = [0.5]×3, std = [0.5]×3)."),
    ("Channel handling", "Grayscale repeated to 3 channels; 2-channel input extended with the first channel; >3 channels truncated to first 3."),
    ("Resampling", "Bilinear for images; nearest-neighbor for masks; masks binarized as value > 0."),
    ("Augmentation", "Recorded as enabled. Training transform sequence: random vertical flip, random horizontal flip, normalization. Flip probabilities are inherited from the custom transform classes and are not surfaced in the training script."),
    ("Objective", "Unweighted sum of binary cross-entropy on probabilities and Dice loss. Dice smoothing = 10⁻⁶."),
    ("Workers", "8 training workers; 1 evaluation worker; pin_memory enabled when CUDA is available (code defaults)."),
])
source(doc, "Source: versamammo_train_seg_v2.py; ZGT training_summary.json. Values marked code default are not run-serialized.")

doc.add_page_break()
doc.add_heading("4. MedSAM LoRA fine-tuning", level=1)
doc.add_paragraph("The checked-in notebook is the executable specification for LoRA fine-tuning on ZGT masks and connected-component bounding-box prompts. No separate run-configuration JSON was found, so the settings below are notebook-defined rather than independently corroborated by a run manifest.")
doc.add_heading("4.1 Backbone, LoRA, and prompts", level=2)
add_table(doc, [
    ("Base model", "MedSAM ViT-B loaded from MedSAM/work_dir/medsam_vit_b.pth."),
    ("PEFT method", "LoRA via Hugging Face PEFT."),
    ("Rank r", "8."),
    ("LoRA alpha", "8 (scaling alpha/r = 1)."),
    ("Target modules", "All modules named qkv matched by PEFT; therefore attention joint query/key/value projections."),
    ("LoRA dropout", "0.1."),
    ("Bias", "none; no bias parameters adapted."),
    ("Prompt encoder", "Explicitly frozen and executed under no_grad for box encoding."),
    ("Other base parameters", "Frozen by PEFT; only injected LoRA parameters are trainable, subject to PEFT module matching."),
    ("Prompt construction", "One box per connected component of the ground-truth mask; annotation threshold 0.5; minimum component area 1 pixel; padding 0; 8-neighborhood connectivity; empty masks allowed."),
    ("Multiple boxes", "Each box is forwarded separately; per-box logits are combined by pixelwise maximum into one union logit map."),
])
source(doc, "Source: pipelines_and_experiments/LoRA_zgt_gtmask_boxes_to_medsam_segmentation.ipynb.")

doc.add_heading("4.2 Optimization and objective", level=2)
add_table(doc, [
    ("Epochs", "50; start_epoch = 0 unless a checkpoint is resumed."),
    ("Batch size", "1."),
    ("Optimizer", "AdamW over parameters with requires_grad = true."),
    ("Learning rate", "1×10⁻⁴."),
    ("Weight decay", "0.01."),
    ("Scheduler / warm-up", "None defined."),
    ("Loss", "Unweighted sum of MONAI DiceLoss and BCEWithLogitsLoss."),
    ("Dice settings", "sigmoid = true; squared_pred = true; reduction = mean; other arguments inherit the installed MONAI defaults."),
    ("BCE settings", "reduction = mean; no class/positive weighting supplied."),
    ("Gradient accumulation / clipping", "None defined."),
    ("Mixed precision", "Not used in the notebook training loop."),
])

doc.add_heading("4.3 Data pipeline, evaluation, and checkpoints", level=2)
add_table(doc, [
    ("Input size", "1,024 × 1,024; grayscale converted to 3 identical channels."),
    ("Image scaling", "Per-image min–max scaling to [0,1] with denominator floor 10⁻⁸; bilinear resize."),
    ("Mask processing", "Binary mask > 0; nearest-neighbor resize to 1,024 × 1,024."),
    ("DataLoader", "shuffle = false; num_workers = 0 for train, evaluation, and test."),
    ("Augmentation", "None defined."),
    ("Evaluation cadence", "Every 5 epochs, including epoch 0."),
    ("Model selection", "Highest mean evaluation Dice; initial best_dice = −1."),
    ("Resume", "resume = true; loads medsam_model_latest.pth if present, including model, optimizer, epoch, best Dice, and histories."),
    ("Checkpointing", "Latest checkpoint saved every epoch; best checkpoint saved when mean evaluation Dice improves."),
    ("Device", "cuda:0 when CUDA is available, otherwise CPU."),
    ("Random seed", "No explicit seed is set in the notebook; exact ordering is deterministic at the loader level but framework/kernel nondeterminism is not controlled."),
])

doc.add_page_break()
doc.add_heading("5. Reproducibility notes and unresolved fields", level=1)
add_note(doc, "Important.", "This record documents what the repository can establish. It should not be read as proof that every planned epoch completed or that code defaults were necessarily used when a run artifact did not serialize the invoked command line.")
add_table(doc, [
    ("nnU-Net completion", "The intended schedule is 1,000 epochs. Debug records capture training state at particular epochs; they are not a compact completion manifest for all five folds."),
    ("nnU-Net library defaults", "LeakyReLU negative slope and some internal loss arguments are not explicit in the plan. Report the framework/library version alongside results if exact reconstruction is required."),
    ("VersaMammo run arguments", "The ZGT training summary does not serialize lr, weight decay, batch sizes, workers, validation cadence, seed, or normalization. This document reports the current script defaults and labels them accordingly."),
    ("VersaMammo flip probability", "The training script instantiates custom horizontal and vertical flip transforms without exposing their probabilities; inspect the transform class version used for the run before claiming an exact probability."),
    ("MedSAM environment", "The notebook does not record PEFT, MONAI, PyTorch, or CUDA versions, nor a seed. These should be added to future run manifests."),
    ("MedSAM split selection", "TEST_DATASET = None selects the fold with best detection mAP@0.50 when metrics are available, otherwise fold 0. The selected fold should be recorded explicitly in reported experiments."),
])

doc.add_heading("6. Evidence index", level=1)
add_table(doc, [
    ("nnU-Net plan", "nnUNet/nnUNet_results/Dataset001_Mammography/nnUNetTrainer__nnUNetPlans__2d/plans.json"),
    ("nnU-Net runtime", "nnUNet/nnUNet_results/Dataset001_Mammography/nnUNetTrainer__nnUNetPlans__2d/fold_0/debug.json"),
    ("nnU-Net split", "nnUNet/nnUNet_preprocessed/Dataset001_Mammography/splits_final.json"),
    ("nnU-Net trainer", "nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer.py"),
    ("nnU-Net LR", "nnUNet/nnunetv2/training/lr_scheduler/polylr.py"),
    ("VersaMammo trainer", "VersaMammo/downstream/Segment/versamammo_train_seg_v2.py"),
    ("VersaMammo run summary", "pipelines_and_experiments/results/SEG_ZGT_versamammo_segmentation/training_summary.json"),
    ("MedSAM LoRA notebook", "pipelines_and_experiments/LoRA_zgt_gtmask_boxes_to_medsam_segmentation.ipynb"),
])

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.core_properties.title = "Training Parameter Record"
doc.core_properties.subject = "nnU-Net, VersaMammo adaptation head, and MedSAM LoRA fine-tuning"
doc.core_properties.author = "FM_thesis reproducibility audit"
doc.save(OUT)
print(OUT.resolve())
