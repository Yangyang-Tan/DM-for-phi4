#!/usr/bin/env wolframscript
(* ::Package:: *)

(* DM-vs-HMC propagator comparison for 2D phi^4 L=128, modeled after
   celeba/GkPlotCelebA128.wls but adapted for the phi^4 lattice pipeline:

   - 2 kappas: {k=0.2705 (near-critical, sigma=450),
                k=0.28   (ordered phase, sigma=640)}
   - 2 samplers: em (SDE), ode (DPM-2)
   - Reference: HMC / wolff+FAHMC training propagator (dashed black)
   - 4 log-spaced epochs chosen to span untrained → converged:
       {0001, 0079, 1291, 10000}
   - Only radial-averaged G(|k|) is available (no x/y/diagonal cuts yet)
   - Log-linear scale: log x, linear y (matches celeba convention)

   Per (k, sampler): 1 raw G(|k|) PDF + 1 ratio PDF = 4 * 2 = 8 PDFs,
   plus one 2x2 composite raw figure and one 2x2 composite ratio figure. *)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

(* ---- configuration ---------------------------------------------------- *)
kappas  = {"0.2705", "0.28"};
sigmas  = {"sigma450", "sigma640"};                 (* parallel to kappas *)
phase   = {"\\text{Near-critical}", "\\text{Ordered}"};
samplers = {"em", "ode"};
epochs  = {"0001", "0079", "1291", "10000"};
epochMX = {"\\text{Epoch 1}", "\\text{Epoch 80}",
           "\\text{Epoch 1291}", "\\text{Epoch 10000}"};

samplerLabel[s_] := Switch[s, "em", "\\mathrm{SDE}", "ode", "\\mathrm{ODE}", _, s];

(* Skip k=0 (log-x needs k>0) *)
loadRaw[f_] := Select[Import[f], (#[[1]] > 0) &];

toAround[rows_]      := {#1, Around[#2, #3]} & @@@ rows;
ratioRows[dmR_, tR_] := MapThread[
   {#1[[1]], Around[#1[[2]]/#2[[2]],
      Sqrt[(#1[[3]]/#2[[2]])^2 + (#1[[2]] #2[[3]]/#2[[2]]^2)^2]]} &,
   {dmR, tR}];

(* Per-(k,sigma) data/figure directories *)
dataDir[i_]   := "sigma_comparison_L128/L128_k" <> kappas[[i]] <> "_" <>
                 sigmas[[i]] <> "/dat_cache/";
figureDir[i_] := "sigma_comparison_L128/L128_k" <> kappas[[i]] <> "_" <>
                 sigmas[[i]] <> "/figures/";

Do[
  dd  = dataDir[i];
  fd  = figureDir[i];
  If[!DirectoryQ[fd], CreateDirectory[fd]];
  Do[
    sLabel = samplerLabel[sampler];
    dmFiles = FileNameJoin[{dd,
        "G_k_DM_" <> sampler <> "_epoch=" <> # <> ".dat"}] & /@ epochs;
    refFile = FileNameJoin[{dd, "G_k_train.dat"}];

    rawDM    = loadRaw /@ dmFiles;
    rawTrain = loadRaw[refFile];

    ratioDatasets = ratioRows[#, rawTrain] & /@ rawDM;
    gkDatasets    = toAround /@ Append[rawDM, rawTrain];

    legendLabels      = MaTeX[#, ContentPadding -> False, FontSize -> 14] & /@ epochMX;
    legendLabelsTrain = MaTeX[#, ContentPadding -> False, FontSize -> 14] & /@
                         Append[epochMX, "\\text{HMC (ref)}"];

    annot = MaTeX["\\kappa = " <> kappas[[i]] <> " /\\; " <> sLabel,
                  ContentPadding -> False, FontSize -> 16];

    (* ---- Raw G(|k|): log-x, linear-y ---- *)
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

    fnameRaw = "prop_phi4_L128_k" <> kappas[[i]] <> "_" <> sampler <> "_epochs.pdf";
    Export[FileNameJoin[{fd, fnameRaw}], plotRaw];
    Print["saved: ", fd <> fnameRaw];

    (* ---- Ratio G_DM/G_train ---- *)
    plotRatio = ListLinePlot[
      ratioDatasets,
      IntervalMarkers -> "Fences",
      IntervalMarkersStyle -> {AbsoluteThickness[1.4], <|"FenceWidth" -> 0.07|>},
      ScalingFunctions -> {"Log", Automatic},
      PlotRange -> {All, {0.7, 1.4}},
      GridLines -> {None, {1}},
      GridLinesStyle -> Directive[Gray, Dashed, Opacity[0.5]],
      PlotLegends -> Placed[
        LineLegend[legendLabels,
         LegendMarkerSize -> {36, 10},
         LegendLayout -> {"Column", 1}],
        Scaled[{0.22, 0.78}]],
      Epilog -> {Inset[annot, Scaled[{0.80, 0.92}], {Right, Top}]},
      FrameLabel -> {MaTeX["|k|", ContentPadding -> False, FontSize -> 15],
                     MaTeX["G_{\\mathrm{DM}}/G_{\\mathrm{train}}",
                           ContentPadding -> False, FontSize -> 15]}];

    fnameRatio = "prop_phi4_L128_k" <> kappas[[i]] <> "_" <> sampler <> "_epochs_ratio.pdf";
    Export[FileNameJoin[{fd, fnameRatio}], plotRatio];
    Print["saved: ", fd <> fnameRatio];
  ,
    {sampler, samplers}];
,
  {i, Length[kappas]}];

(* ------------------------------------------------------------------------
   2x2 composite panels (rows = kappa, cols = sampler), raw + ratio.
   ------------------------------------------------------------------------ *)
makeRawPanel[i_, sampler_] := Module[
  {dd, sLabel, dmFiles, refFile, rawDM, rawTrain, gkDatasets, legendLabelsT, annot},
  dd = dataDir[i];
  sLabel = samplerLabel[sampler];
  dmFiles = FileNameJoin[{dd,
      "G_k_DM_" <> sampler <> "_epoch=" <> # <> ".dat"}] & /@ epochs;
  refFile = FileNameJoin[{dd, "G_k_train.dat"}];
  rawDM    = loadRaw /@ dmFiles;
  rawTrain = loadRaw[refFile];
  gkDatasets = toAround /@ Append[rawDM, rawTrain];
  legendLabelsT = MaTeX[#, ContentPadding -> False, FontSize -> 10] & /@
                   Append[epochMX, "\\text{HMC}"];
  annot = MaTeX["\\kappa = " <> kappas[[i]] <> " /\\; " <> sLabel,
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

makeRatioPanel[i_, sampler_] := Module[
  {dd, sLabel, dmFiles, refFile, rawDM, rawTrain, ratioDatasets, legendLabels, annot},
  dd = dataDir[i];
  sLabel = samplerLabel[sampler];
  dmFiles = FileNameJoin[{dd,
      "G_k_DM_" <> sampler <> "_epoch=" <> # <> ".dat"}] & /@ epochs;
  refFile = FileNameJoin[{dd, "G_k_train.dat"}];
  rawDM    = loadRaw /@ dmFiles;
  rawTrain = loadRaw[refFile];
  ratioDatasets = ratioRows[#, rawTrain] & /@ rawDM;
  legendLabels  = MaTeX[#, ContentPadding -> False, FontSize -> 11] & /@ epochMX;
  annot = MaTeX["\\kappa = " <> kappas[[i]] <> " /\\; " <> sLabel,
                ContentPadding -> False, FontSize -> 13];
  ListLinePlot[
    ratioDatasets,
    IntervalMarkers -> "Fences",
    IntervalMarkersStyle -> {AbsoluteThickness[1.0], <|"FenceWidth" -> 0.05|>},
    ScalingFunctions -> {"Log", Automatic},
    PlotRange -> {All, {0.7, 1.4}},
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

compositeDir = "sigma_comparison_L128/figures/";
If[!DirectoryQ[compositeDir], CreateDirectory[compositeDir]];

rawComposite = Grid[
  Table[makeRawPanel[i, sampler],
    {i, Length[kappas]}, {sampler, samplers}],
  Spacings -> {2, 2}, Frame -> None];
ratioComposite = Grid[
  Table[makeRatioPanel[i, sampler],
    {i, Length[kappas]}, {sampler, samplers}],
  Spacings -> {2, 2}, Frame -> None];

Export[FileNameJoin[{compositeDir, "prop_phi4_L128_2x2_raw.pdf"}],    rawComposite];
Export[FileNameJoin[{compositeDir, "prop_phi4_L128_2x2_ratio.pdf"}], ratioComposite];
Print["saved: ", compositeDir <> "prop_phi4_L128_2x2_{raw,ratio}.pdf"];
