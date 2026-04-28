#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Six-panel propagator comparison: {x, y, diagonal} × {em/SDE, dpm2/ODE}.
   For each combo, produces one ratio PDF (G_DM/G_train vs |k|) with 4
   well-separated epochs chosen for maximum visual distinguishability. *)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

dataDir   = "celeba_128_ncsnpp/correlation/";
figureDir = "celeba_128_ncsnpp/figures/";

(* Four log-spaced epochs spanning untrained → fully-converged, chosen for
   max separation in the ratio plot. The final checkpoint lives at
   ep=9999 on disk but represents 10000 completed epochs (Lightning saves
   at the end of epoch N, so epoch=9999 is the 10000th trained epoch);
   em and dpm2 10k-sample files exist for all four epochs. *)
epochs  = {"0001", "0099", "0899", "9999"};
epochMX = {"\\text{Epoch 1}", "\\text{Epoch 100}",
           "\\text{Epoch 900}", "\\text{Epoch 10000}"};

(* Skip only k=0 (log-x requires k>0); keep full k range otherwise. *)
loadRaw[f_] := Select[Import[f], (#[[1]] > 0) &];

toAround[rows_]       := {#1, Around[#2, #3]} & @@@ rows;
ratioRows[dmR_, tR_]  := MapThread[
   {#1[[1]], Around[#1[[2]]/#2[[2]],
      Sqrt[(#1[[3]]/#2[[2]])^2 + (#1[[2]] #2[[3]]/#2[[2]]^2)^2]]} &,
   {dmR, tR}];

(* Map direction symbol → file-name suffix and display name *)
dirSuffix[d_] := "_" <> d;
dirPretty[d_] := Switch[d,
  "x",        "X",
  "y",        "Y",
  "diagonal", "Diagonal",
  "all",      "All",
  _,          d];

(* Sampler display names for in-plot annotation *)
samplerLabel[s_] := Switch[s, "em", "\\mathrm{SDE}", "dpm2", "\\mathrm{ODE}", _, s];

(* Directions plotted. `all` = full angular average (max_diagonality=1)
   over every lattice mode; x, y, diagonal are axial slices. *)
directions = {"x", "y", "diagonal", "all"};

(* -----------------------------------------------------------------
   For each (direction, sampler): two PDFs — raw G(|k|) vs |k| and
   ratio G_DM/G_train vs |k|. Raw uses log-x / linear-y (full k-range).
   No PlotLabel; direction+sampler annotated inside the frame.
   8 directions × 2 samplers × 2 styles = 16 PDFs, written to figures/.
   ----------------------------------------------------------------- *)
Do[
  dirSfx = dirSuffix[direction];
  sLabel = samplerLabel[sampler];
  dPretty = dirPretty[direction];

  dmFiles = FileNameJoin[{dataDir,
      "G_k_DM_celeba_128_" <> sampler <> dirSfx <>
      "_epoch=" <> # <> ".dat"}] & /@ epochs;
  refFile = FileNameJoin[{dataDir,
      "G_k_train_celeba_128" <> dirSfx <> ".dat"}];

  rawDM    = loadRaw /@ dmFiles;
  rawTrain = loadRaw[refFile];

  ratioDatasets = ratioRows[#, rawTrain] & /@ rawDM;
  gkDatasets    = toAround /@ Append[rawDM, rawTrain];

  legendLabels      = MaTeX[#, ContentPadding -> False, FontSize -> 14] & /@ epochMX;
  legendLabelsTrain = MaTeX[#, ContentPadding -> False, FontSize -> 14] & /@
                       Append[epochMX, "\\text{CelebA HMC}"];

  (* In-frame annotation "Direction / Sampler" replaces PlotLabel *)
  annot = MaTeX["\\text{" <> dPretty <> " / } " <> sLabel,
                ContentPadding -> False, FontSize -> 16];

  (* ---- Raw G(|k|): log-x, linear-y, 4 epochs + training ---- *)
  plotRaw = ListLinePlot[
    gkDatasets,
    IntervalMarkers -> "Fences",
    IntervalMarkersStyle -> <|"FenceWidth" -> 0.06, "FenceStyle" -> Thickness[0.003]|>,
    ScalingFunctions -> {"Log", Automatic},
    PlotRange -> All,
    PlotStyle -> {Automatic, Automatic, Automatic, Automatic,
       Directive[Black, Dashed]},
    PlotLegends -> Placed[
      LineLegend[legendLabelsTrain,
       LegendMarkerSize -> {36, 10},
       LegendLayout -> {"Column", 1}],
      Scaled[{0.78, 0.70}]],
    Epilog -> {Inset[annot, Scaled[{0.20, 0.92}], {Left, Top}]},
    FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 15],
                   MaTeX["G(|k|)", ContentPadding -> False, FontSize -> 15]}];

  fnameRaw = "prop_celeba_128_" <> sampler <> dirSfx <> "_epochs.pdf";
  Export[FileNameJoin[{figureDir, fnameRaw}], plotRaw];
  Print["saved: ", fnameRaw];

  (* ---- Ratio plot ---- *)
  plotRatio = ListLinePlot[
    ratioDatasets,
    IntervalMarkers -> "Fences",
    IntervalMarkersStyle -> {AbsoluteThickness[1.4], <|"FenceWidth" -> 0.07|>},
    ScalingFunctions -> {"Log", Automatic},
    PlotRange -> {All, {0.55, 1.7}},
    GridLines -> {None, {1}},
    GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.5]],
    PlotLegends -> Placed[
      LineLegend[legendLabels,
       LegendMarkerSize -> {36, 10},
       LegendLayout -> {"Column", 1}],
      Scaled[{0.20, 0.78}]],
    Epilog -> {Inset[annot, Scaled[{0.80, 0.92}], {Right, Top}]},
    FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 15],
                   MaTeX["G_{\\mathrm{DM}}/G_{\\mathrm{train}}",
                         ContentPadding -> False, FontSize -> 15]}];

  fnameRatio = "prop_celeba_128_" <> sampler <> dirSfx <> "_epochs_ratio.pdf";
  Export[FileNameJoin[{figureDir, fnameRatio}], plotRatio];
  Print["saved: ", fnameRatio];
,
  {sampler, {"em", "dpm2"}},
  {direction, directions}];

(* -----------------------------------------------------------------
   Two composites (2 rows × 4 cols = 8 panels each): raw and ratio.
   ----------------------------------------------------------------- *)
makeRawPanel[sampler_, direction_] := Module[
  {dirSfx, sLabel, dPretty, dmFiles, refFile, rawDM, rawTrain,
   gkDatasets, legendLabelsT, annot},
  dirSfx  = dirSuffix[direction];
  sLabel  = samplerLabel[sampler];
  dPretty = dirPretty[direction];
  dmFiles = FileNameJoin[{dataDir,
      "G_k_DM_celeba_128_" <> sampler <> dirSfx <>
      "_epoch=" <> # <> ".dat"}] & /@ epochs;
  refFile = FileNameJoin[{dataDir,
      "G_k_train_celeba_128" <> dirSfx <> ".dat"}];
  rawDM    = loadRaw /@ dmFiles;
  rawTrain = loadRaw[refFile];
  gkDatasets = toAround /@ Append[rawDM, rawTrain];
  legendLabelsT = MaTeX[#, ContentPadding -> False, FontSize -> 10] & /@
                   Append[epochMX, "\\text{CelebA HMC}"];
  annot = MaTeX["\\text{" <> dPretty <> " / } " <> sLabel,
                ContentPadding -> False, FontSize -> 13];
  ListLinePlot[
    gkDatasets,
    IntervalMarkers -> "Fences",
    IntervalMarkersStyle -> <|"FenceWidth" -> 0.05, "FenceStyle" -> Thickness[0.002]|>,
    ScalingFunctions -> {"Log", Automatic},
    PlotRange -> All,
    PlotStyle -> {Automatic, Automatic, Automatic, Automatic,
       Directive[Black, Dashed]},
    PlotLegends -> Placed[
      LineLegend[legendLabelsT,
       LegendMarkerSize -> {22, 6},
       LegendLayout -> {"Column", 1}],
      Scaled[{0.80, 0.72}]],
    Epilog -> {Inset[annot, Scaled[{0.22, 0.92}], {Left, Top}]},
    FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 12],
                   MaTeX["G(|k|)", ContentPadding -> False, FontSize -> 12]},
    ImageSize  -> 360]];

makeRatioPanel[sampler_, direction_] := Module[
  {dirSfx, sLabel, dPretty, dmFiles, refFile, rawDM, rawTrain,
   ratioDatasets, legendLabels, annot},
  dirSfx  = dirSuffix[direction];
  sLabel  = samplerLabel[sampler];
  dPretty = dirPretty[direction];
  dmFiles = FileNameJoin[{dataDir,
      "G_k_DM_celeba_128_" <> sampler <> dirSfx <>
      "_epoch=" <> # <> ".dat"}] & /@ epochs;
  refFile = FileNameJoin[{dataDir,
      "G_k_train_celeba_128" <> dirSfx <> ".dat"}];
  rawDM    = loadRaw /@ dmFiles;
  rawTrain = loadRaw[refFile];
  ratioDatasets = ratioRows[#, rawTrain] & /@ rawDM;
  legendLabels  = MaTeX[#, ContentPadding -> False, FontSize -> 11] & /@ epochMX;
  annot = MaTeX["\\text{" <> dPretty <> " / } " <> sLabel,
                ContentPadding -> False, FontSize -> 13];
  ListLinePlot[
    ratioDatasets,
    IntervalMarkers -> "Fences",
    IntervalMarkersStyle -> {AbsoluteThickness[1.0], <|"FenceWidth" -> 0.05|>},
    ScalingFunctions -> {"Log", Automatic},
    PlotRange -> {All, {0.55, 1.7}},
    GridLines -> {None, {1}},
    GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.5]],
    PlotLegends -> Placed[
      LineLegend[legendLabels,
       LegendMarkerSize -> {26, 7},
       LegendLayout -> {"Column", 1}],
      Scaled[{0.22, 0.80}]],
    Epilog -> {Inset[annot, Scaled[{0.80, 0.92}], {Right, Top}]},
    FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 12],
                   MaTeX["G_{\\mathrm{DM}}/G_{\\mathrm{train}}",
                         ContentPadding -> False, FontSize -> 12]},
    ImageSize  -> 360]];

rawComposite = Grid[
  Table[makeRawPanel[sampler, direction],
    {sampler, {"em", "dpm2"}}, {direction, directions}],
  Spacings -> {2, 2}, Frame -> None];
ratioComposite = Grid[
  Table[makeRatioPanel[sampler, direction],
    {sampler, {"em", "dpm2"}}, {direction, directions}],
  Spacings -> {2, 2}, Frame -> None];

Export[FileNameJoin[{figureDir, "prop_celeba_128_8panel_epochs.pdf"}],       rawComposite];
Export[FileNameJoin[{figureDir, "prop_celeba_128_8panel_epochs_ratio.pdf"}], ratioComposite];
Print["saved: prop_celeba_128_8panel_epochs{,_ratio}.pdf"];
