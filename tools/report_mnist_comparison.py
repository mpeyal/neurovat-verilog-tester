"""Build the Word report from recorded measurements and rendered figures."""
import json
from pathlib import Path
from datetime import date
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path(__file__).resolve().parents[1]/"results"/"mnist_ecfet_v2"


def main():
    d = json.loads((OUT/"comparison.json").read_text())
    gui = json.loads((OUT/"desktop_verification.json").read_text())
    manifest = json.loads((OUT/"data_manifest.json").read_text())
    activity = json.loads((OUT/"activity_analysis.json").read_text())
    h, final, epoch = d["history"], d["final"], d["best"]["app"][2]
    errors = {key: max(row["errors"][key] for row in h) for key in
              ("max_gradient_error", "max_loss_error", "max_weight_error_S")}
    mismatches = sum(row["errors"]["spike_mismatches"] for row in h)
    eval_mismatch = sum(row[s]["spike_mismatches"] for row in h for s in final)
    eval_disagree = sum(row[s]["prediction_disagreements"] for row in h for s in final)
    max_mem = max(row[s]["max_membrane_error"] for row in h for s in final)
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.5), Inches(11)
    sec.top_margin, sec.bottom_margin = Inches(.65), Inches(.65)
    sec.left_margin, sec.right_margin = Inches(.8), Inches(.8)
    sec.header_distance, sec.footer_distance = Inches(.25), Inches(.3)
    for name in ("Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Caption"):
        style = doc.styles[name]
        style.font.name = "Calibri"
        style.font.color.rgb = RGBColor(0, 0, 0)
    normal = doc.styles["Normal"]
    normal.font.size = Pt(11)
    normal.paragraph_format.space_after = Pt(7)
    normal.paragraph_format.line_spacing = 1.08
    doc.styles["Title"].font.size = Pt(25)
    for border in list(doc.styles["Title"]._element.xpath("./w:pPr/w:pBdr")):
        border.getparent().remove(border)
    doc.styles["Heading 1"].font.size = Pt(17)
    doc.styles["Heading 1"].paragraph_format.space_before = Pt(10)
    doc.styles["Heading 2"].font.size = Pt(12)
    doc.styles["Caption"].font.size = Pt(9)
    doc.styles["Caption"].font.bold = False
    doc.styles["Caption"].paragraph_format.space_after = Pt(8)
    header = sec.header.paragraphs[0]
    header.text = ""
    footer = sec.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    field = OxmlElement("w:fldSimple"); field.set(qn("w:instr"), "PAGE")
    footer._p.append(field)

    def p(text, style=None): return doc.add_paragraph(text, style)
    def head(text): doc.add_heading(text, level=1)
    def page(text): doc.add_page_break(); head(text)
    def fig(name, caption, width=6.8):
        para = doc.add_paragraph(); para.paragraph_format.space_after = Pt(2)
        para.add_run().add_picture(str(OUT/f"{name}.png"), width=Inches(width))
        p(caption, "Caption")
    def table(headers, rows, widths):
        tab = doc.add_table(rows=1, cols=len(headers)); tab.autofit = False
        for col, width in zip(tab.columns, widths): col.width = Inches(width)
        for i, text in enumerate(headers): tab.rows[0].cells[i].text = text
        for values in rows:
            cells = tab.add_row().cells
            for cell, text in zip(cells, values): cell.text = str(text)
        for ri, row in enumerate(tab.rows):
            for ci, cell in enumerate(row.cells):
                cell.width = Inches(widths[ci]); cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                tcpr = cell._tc.get_or_add_tcPr()
                borders = OxmlElement("w:tcBorders")
                for side in ("top", "left", "bottom", "right"):
                    edge = OxmlElement(f"w:{side}"); edge.set(qn("w:val"), "single")
                    edge.set(qn("w:sz"), "4"); edge.set(qn("w:color"), "D9D9D9"); borders.append(edge)
                tcpr.append(borders)
                margins = OxmlElement("w:tcMar")
                for side in ("top", "left", "bottom", "right"):
                    margin = OxmlElement(f"w:{side}"); margin.set(qn("w:w"), "85")
                    margin.set(qn("w:type"), "dxa"); margins.append(margin)
                tcpr.append(margins)
                if ri == 0:
                    fill = OxmlElement("w:shd"); fill.set(qn("w:fill"), "E1EAF1"); tcpr.append(fill)
                for para in cell.paragraphs:
                    para.paragraph_format.space_after = Pt(1)
                    para.paragraph_format.line_spacing = 1.0
                    if ci > 0: para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    for run in para.runs:
                        run.font.size = Pt(10)
                        run.bold = ri == 0
            if ri == 0:
                repeat = OxmlElement("w:tblHeader"); row._tr.get_or_add_trPr().append(repeat)
        p("").paragraph_format.space_after = Pt(0)

    doc.add_paragraph("MNIST validation of ECFET v2 training", "Title")
    p(f"NeuroVAT app engine and matched snnTorch reference\nMeasured experiment • {date.today().strftime('%d %B %Y')}", "Subtitle")
    matched = eval_disagree == 0 and eval_mismatch == 0 and mismatches == 0
    p(("The two implementations agreed on every recorded prediction and output spike. " if matched else
       "The comparison found differences between the implementations. ")+
      f"At the independently selected checkpoints, the app achieved {final['train']['app_accuracy']:.1%} training accuracy and "
      f"{final['test']['app_accuracy']:.1%} test accuracy; snnTorch achieved {final['train']['snntorch_accuracy']:.1%} and "
      f"{final['test']['snntorch_accuracy']:.1%}, respectively. These are measured results from real MNIST images.")
    p("This experiment checks software consistency for one ECFET v2 configuration using surrogate-gradient learning. "
      "It uses a balanced subset of MNIST and one random seed. It does not establish full-dataset performance or validate physical device behavior.")
    p(f"A significant limitation is output saturation: neurons fire in {activity['occupancy']:.1%} of test time bins. "
      f"The reported {final['test']['app_accuracy']:.1%} test accuracy uses membrane values; a diagnostic spike-count readout reaches only "
      f"{activity['spike_count_argmax_accuracy']:.1%}. Agreement therefore supports software consistency, but not useful temporal coding at these settings.")
    head("Selected checkpoint results")
    table(["Metric", "App", "snnTorch"], [
        ["Selected epoch", epoch, d["best"]["snntorch"][2]],
        *[[f"{s.title()} accuracy  n = {d['counts'][s]:,}", f"{final[s]['app_accuracy']:.1%}", f"{final[s]['snntorch_accuracy']:.1%}"] for s in final]], [3.5, 1.65, 1.65])
    p(f"The selected checkpoints differ on {final['test']['prediction_disagreements']} of 500 test predictions "
      f"and {final['test']['spike_mismatches']} of 300,000 output spike bins.")
    head("Experiment at a glance")
    p("Ten classes, digits 0–9. Each class contributes 100 training, 20 validation and 50 test images. "
      "Images are resized from 28 × 28 to 14 × 14 using the app’s bilinear preprocessing. "
      "The network has 196 input channels and 10 output LIF neurons, with 1,960 ECFET v2 synapses and no hidden layer.")
    p("Both runs use identical initial device states, image order and replayed Poisson spike trains. "
      "Training comprises eight epochs of 1,000 presentations. Each implementation independently selects the highest validation accuracy, "
      "breaking ties using lower training loss. Test images are evaluated only after training and selection have finished.")

    page("Training and test accuracy")
    fig("accuracy_comparison", "Figure 1. Accuracy on fixed spike banks. Overlaid blue circles and orange crosses denote app and snnTorch. "
        "The dotted line marks the app’s selected epoch. Test curves were calculated after checkpoint selection was frozen.")
    table(["Epoch", "Train %", "Validation %", "Test %"],
          [[r["epoch"], *[f"{100*r[s]['app_accuracy']:.1f} / {100*r[s]['snntorch_accuracy']:.1f}" for s in final]] for r in h],
          [.65, 2.05, 2.05, 2.05])
    p("Each cell lists app / snnTorch. Training accuracy is a separate forward-only evaluation after each epoch, "
      "using a fixed spike realization; it is not the fraction correct during weight updates. Epoch 0 is the untrained baseline.")
    fig("loss_comparison", "Figure 2. Mean online cross-entropy loss across the 1,000 presentations in each epoch. "
        "The two loss traces overlap at the plotted scale.", width=6.3)

    page("Spike timing agreement")
    fig("spike_raster_detail", "Figure 3. The first selected test image of digit 0, its shared input spike train, "
        "both output rasters on the same axes, and the predicted neuron’s post-reset membrane. "
        "Blue ticks and orange dots are displaced vertically by 0.14 neuron units only to reveal overlap.")
    p("The input encoder emits binary events on a 1 ms grid, with pixel intensity controlling the event probability "
      "up to 160 Hz. A presentation lasts 60 ms. These events drive a computational LIF network; they are not measured "
      "ECFET gate-voltage waveforms. The neuron membrane provides temporal integration.")
    p("Classification uses the largest mean pre-reset output membrane value. Consequently, the predicted digit need not "
      "be the neuron with the largest spike count. The lower trace shows the post-reset membrane for visualization, "
      "whereas the classifier reads the pre-reset values.")
    p(f"Across the nine checkpoint evaluations, there were {eval_disagree:,} prediction disagreements in 15,300 image evaluations "
      f"and {eval_mismatch:,} differences in 9,180,000 output spike bins. Across the 8,000 training presentations, "
      f"there were {mismatches:,} differences in 4,800,000 output spike bins. The largest evaluated membrane difference was {max_mem:.3e} arbitrary units.")

    page("Paired rasters for all ten digits")
    fig("spike_raster_all_digits", "Figure 4. First selected test image of each digit, chosen by fixed index rather than prediction quality. "
        "Predictions are listed as app / snnTorch. Insets show the actual 14 × 14 inputs. "
        "The standalone PNG and SVG permit close inspection.", width=6.5)

    page("Recognition errors and numerical agreement")
    fig("test_confusion", "Figure 5. Confusion matrices at each implementation’s validation-selected checkpoint. "
        "Each true class has 50 test images. Values are counts, not percentages.")
    table(["Comparison across training", "Maximum absolute difference"], [
        ["Cross-entropy loss", f"{errors['max_loss_error']:.3e}"],
        ["Gradient of normalized synaptic weight", f"{errors['max_gradient_error']:.3e}"],
        ["Physical conductance during training", f"{errors['max_weight_error_S']:.3e} S"],
        ["Conductance at selected checkpoints", f"{d['final_max_weight_error_S']:.3e} S"]], [4.0, 2.8])
    p("The measured errors quantify numerical agreement; there is no claim of bitwise identity. "
      "Both implementations follow the same device-limited update rule. The reference computes its own neuron trajectories "
      "and gradients with snnTorch and PyTorch autograd, then applies them to a separate copy of the ECFET array.")
    head("Desktop integration check")
    p(f"The native DearPyGui app built the network, saved and reloaded the complete selected checkpoint, and ran its "
      f"asynchronous Evaluate worker. With the same saved input spikes, it displayed {gui['train_accuracy']:.1%} training "
      f"and {gui['test_accuracy']:.1%} test accuracy. All ten example output rasters matched the backend records. "
      "The eight-epoch comparison itself used the app training engine directly; it was not a frame-by-frame GUI training run.")
    p("The supplied model can be opened through Network Lab’s Load control. Its normal interactive encoder creates the app’s "
      "usual deterministic evaluation spikes, which can differ from this experiment’s per-image replay bank. "
      "Use the supplied desktop verification script to reproduce these exact accuracy values.")

    page("Methods and reproducibility")
    table(["Setting", "Value in both runs"], [
        ["LIF membrane time constant and threshold", "20 ms; 1.0 arbitrary unit"],
        ["Decay and reset", "exp(−1/20); immediate subtraction"],
        ["Input normalization and gain", "Divide by active inputs + 1; gain 11"],
        ["Learning rule", "BPTT; fast sigmoid slope 5"],
        ["Rate schedule", "0.1 × cosine scale from 1.0 to 0.15"],
        ["Programming amplitudes and base width", "50 pA; 10 ms"],
        ["ECFET resistance limits", "870 Ω to 10,000 Ω"],
        ["Intrinsic STDP and write noise", "A_stdp = 0; sigma_c2c = 0"],
        ["Random seed", "17"],
        ["Library versions", f"PyTorch {d['settings']['torch']}; snnTorch {d['settings']['snntorch']}"]], [3.2, 3.6])
    p("The app computes BPTT manually; the reference uses snnTorch Leaky neurons and a float64-preserving wrapper "
      "around its FastSigmoid surrogate. The reference disables the built-in reset and performs an explicit differentiable "
      "immediate subtraction to match the app. Neither run uses ordinary Adam or unconstrained SGD updates: gradients "
      "are converted into the app’s conductance-programming pulses, including its update threshold and device bounds.")
    p("This surrogate mode does not apply the STDP mode’s refractory, homeostatic, teacher or lateral-inhibition rules. "
      "ECFET v2 stores conductance; this experiment does not require the device to remember spike timestamps. "
      "It does not test physical STDP waveform shaping, RC circuitry, Spectre or Verilog-A simulation.")
    head("Files and repeat commands")
    p("All outputs are in results/mnist_ecfet_v2. comparison.json contains predictions and raster samples; "
      "accuracy.csv contains epoch accuracies; data_manifest.json records exact source indices and the archive hash. "
      "prepared_data.npz, evaluation_spikes.npz and both selected checkpoint files preserve the experiment inputs and states.")
    for command in ("python tools/mnist_ecfet_v2.py", "python tools/verify_mnist_desktop.py", "python tools/analyze_mnist_activity.py", "python tools/plot_mnist_comparison.py"):
        para = p(command); para.paragraph_format.space_after = Pt(3)
        para.runs[0].font.name = "Consolas"; para.runs[0].font.size = Pt(9)
    p("The Word builder is tools/report_mnist_comparison.py. Running the experiment again overwrites this output directory’s "
      "experiment files. Dependencies include requirements-validation.txt and the app requirements.")

    page("Output saturation and readout limits")
    fig("activity_readout", "Figure 6. Activity measured over all 500 test images at the selected checkpoint. "
        "The spike-count readout is a post-training diagnostic; it was not used for training or checkpoint selection. "
        "snnTorch has identical output events on these images.")
    table(["Output activity measurement", "Result"], [
        ["Spikes across 300,000 available bins", f"{activity['output_spikes']:,}  ({activity['occupancy']:.1%})"],
        ["Mean output firing rate", f"{activity['mean_output_rate_hz']:.1f} Hz"],
        ["Neuron presentations firing in all 60 bins", f"{activity['neuron_presentations_firing_every_step']:,} / 5,000"],
        ["Images with all ten neurons firing in all bins", f"{activity['images_with_all_neurons_firing_every_step']} / 500"],
        ["Images with a tie for greatest spike count", f"{activity['spike_count_argmax_tied_images']} / 500"],
        ["Largest post-reset membrane value", f"{activity['max_post_reset_membrane']:.3f} a.u.  (threshold 1)"]], [4.4, 2.4])
    p("At 1 ms resolution the implementation emits at most one spike per neuron per step, giving a 1,000 Hz ceiling. "
      "The current gain drives most outputs close to that ceiling. An immediate subtraction removes one threshold per step; "
      "it does not force an overdriven membrane below threshold. This explains the dense raster and the elevated membrane trace.")
    p("The network learns discriminative membrane values, but these settings provide little separation in spike counts "
      "and weak evidence of informative spike timing. The diagnostic count classifier chooses the lowest neuron index when "
      "counts tie, so its 21.8% accuracy depends on that declared tie rule. It is not a separately trained spike-count model.")
    p("Before presenting this as a timing-based SNN result, tune input drive using training and validation data, examine firing "
      "rates, and choose a training objective consistent with the intended spike or membrane readout. Then freeze those choices "
      "before a new held-out evaluation. No gain retuning or new training was performed after this test result was observed.")

    page("Interpretation and sources")
    head("What this result supports")
    p("For this selected architecture, dataset subset and seed, the app’s surrogate-gradient training behavior agrees with "
      "a separately computed snnTorch neuron and autograd reference. The accuracy improvements over the untrained baseline "
      "come from real supervised updates to ECFET conductance states. The paired rasters provide an event-level comparison "
      "in addition to aggregate accuracy. The observed saturation means this run should be described as membrane-readout "
      "classification with matched spiking dynamics, rather than a demonstration of effective spike-timing classification.")
    head("What remains untested")
    p("The same ECFET programming code is used on both sides, so a shared device-model error would not be detected. "
      "Physical device measurements or an independent circuit model are needed for that claim. A full 60,000-image training "
      "and 10,000-image test benchmark, repeated seeds, a hidden-layer model and robustness to write noise were not run here. "
      "The measured subset accuracy should not be presented as full-MNIST performance.")
    p("Numerical agreement is also conditional on matching the encoder, reset timing, readout, precision and update rule. "
      "A standard snnTorch tutorial network with a different architecture or Adam optimizer would be a different experiment, "
      "and its results would not be expected to match these curves.")
    head("Data and implementation provenance")
    p("The official MNIST archive supplies 60,000 training and 10,000 test images [1]. The experiment draws validation "
      "images only from the official training split and test images only from the official test split; training and validation indices do not overlap.")
    p("Archive SHA256", "Caption")
    para = p(manifest["sha256"])
    para.runs[0].font.name = "Consolas"; para.runs[0].font.size = Pt(8)
    p("Local primary evidence: tools/mnist_ecfet_v2.py; tools/validate_snntorch.py; vatester/neuro.py; simulation_core/model_v2.py; "
      "comparison.json; desktop_verification.json. Numerical claims in this report are derived from those recorded outputs.")
    for title, url in (("1  Keras MNIST dataset documentation", "https://keras.io/api/datasets/mnist/"),
                       ("2  snnTorch Leaky neuron documentation", "https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html"),
                       ("3  snnTorch surrogate gradient documentation", "https://snntorch.readthedocs.io/en/stable/snntorch.surrogate.html")):
        para = p(title); para.paragraph_format.space_after = Pt(2)
        para.runs[0].bold = True
        para = p(url); para.runs[0].font.size = Pt(9)
    doc.core_properties.title = "MNIST validation of ECFET v2 training"
    doc.core_properties.subject = "Measured comparison of NeuroVAT and matched snnTorch training"
    doc.core_properties.author = "NeuroVAT validation"
    target = OUT/"MNIST_ECFET_v2_validation_report.docx"
    doc.save(target)
    print(target)


if __name__ == "__main__":
    main()
