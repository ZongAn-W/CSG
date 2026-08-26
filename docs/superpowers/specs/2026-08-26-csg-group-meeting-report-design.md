# CSG Group Meeting Report Design

## Objective

Create a polished Chinese Word report for an internal research group meeting.
The report explains the CSG model implemented in
`models/convlstm_ls_topography_joint_gated_simvp.py`, presents the supplied
performance figures, and separates demonstrated implementation properties from
claims that still require controlled experiments.

## Audience And Use

The primary audience is a technically literate research group familiar with
spatiotemporal forecasting. The document should work both as a standalone
meeting handout and as speaking notes. It will use concise technical prose,
explicit tensor shapes, and one central equation without becoming a paper-style
derivation.

## Deliverable

- One Chinese `.docx` file, approximately 6-8 pages.
- A clean research-report visual system using white space, dark text, and blue,
  orange, and green accents drawn from the supplied charts.
- The three supplied images embedded with numbered captions and explanatory
  discussion.
- No invented author, institution, training configuration, dataset split, or
  unshown metric values. Missing administrative details are omitted rather than
  represented by visible placeholders.

## Narrative Structure

1. **Cover and executive summary**: report title, date, research question, and
   the main result visible in the figures.
2. **Problem and motivation**: explain why ozone forecasting may benefit from
   temporal dynamics, solar longitude, and static MOLA topography.
3. **Baselines and CSG architecture**: distinguish SimVP, ConvLSTM-SimVP, and
   CSG; show the model pipeline and input/output tensor contracts.
4. **Conditional spatiotemporal gate**: explain the Ls harmonic encoder,
   topography encoder, joint nonlinear fusion, bounded residual scaling, and
   near-identity initialization.
5. **Experimental results**: interpret overall RMSE and the 20-step RMSE curves
   qualitatively. Any values read from screenshots are described as approximate
   unless explicitly labelled in the source figure.
6. **Engineering validation**: report parameter count, relative overhead,
   shape/backpropagation coverage, and test results obtained directly from the
   repository.
7. **Discussion and next steps**: identify the multiplicative-only conditioning
   limitation, dtype inconsistency, conservative initial gate effect, and the
   need for multi-seed ablations and stratified evaluation.

## Figure Plan

- **Figure 1, architecture**: use the supplied CSG flow diagram near the method
  overview. The prose will clarify that the implementation contains one main
  ConvLSTM-SimVP path plus three inputs to the joint gate.
- **Figure 2, overall comparison**: use the supplied model comparison bar chart
  in the results section. Emphasize ranking and relative improvement rather than
  overstating screenshot-derived precision.
- **Figure 3, forecast-step behavior**: use the supplied 20-step RMSE plot after
  the overall comparison. Discuss the widening long-horizon separation and the
  shared saturation trend.

## Content Accuracy Rules

- CSG refers to the supplied implementation file, not to a broader published
  architecture unless a citation is provided later.
- The current model uses historical Ls only and static topography without a
  time axis.
- The gate output is multiplicative: `E_gated = E * (1 + s * gate)`.
- Unit tests establish interface, shape, gradient, and regression behavior;
  they do not establish scientific superiority.
- The screenshots support qualitative model comparisons. They do not establish
  statistical significance, training fairness, or reproducibility.

## Layout And QA

Use A4 portrait pages, a restrained cover, running header/footer, numbered
sections, compact callout boxes, and figure captions kept with their images.
Images must remain legible without cropping or distortion. The final DOCX will
be rendered page by page and inspected for clipping, overlap, awkward page
breaks, missing glyphs, and unreadable chart labels before delivery.

