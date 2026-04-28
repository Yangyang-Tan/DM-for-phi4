#!/usr/bin/env wolframscript
(* ::Package:: *)

(* Redraws the FID vs propagator-metric comparison figure in Mathematica,
   matching the Python prototype celeba/celeba_128_ncsnpp/figures/fid_dtot_l1_rvar_celeba_128.{png,pdf}.

   Produces 6 single-panel PDFs and a 2x3 composite, written to
   celeba_128_ncsnpp/figures/. Uses the same data sources:
     - FID CSVs produced by compute_fid_celeba.py
     - "All"-direction propagators from correlation_celeba.jl
       (max_diagonality=1.0, full angular average)
*)

SetDirectory[DirectoryName[If[$InputFileName === "", NotebookFileName[], $InputFileName]]];
Get[FileNameJoin[{"..", "academic.m"}]];
Needs["MaTeX`"];

(* ============================================================ *)
(*                  Data paths & constants                      *)
(* ============================================================ *)

dataDir   = "celeba_128_ncsnpp";
corrDir   = FileNameJoin[{dataDir, "correlation"}];
figureDir = FileNameJoin[{dataDir, "figures"}];

epochs    = {1, 2, 4, 6, 11, 18, 29, 47, 99, 149, 199, 349, 549, 899,
             1449, 2349, 3799, 6149, 9999};
epochsStr = IntegerString[epochs, 10, 4];
dim       = 2;                 (* 2D lattice *)

(* ============================================================ *)
(*                       Loaders                                *)
(* ============================================================ *)

(* FID CSV: header + rows "epoch, n_gen, fid, path" *)
loadFid[method_String] := Module[{rows},
  rows = Rest @ Import[FileNameJoin[{dataDir, "fid_" <> method <> "_N2048.csv"}]];
  (* drop k=0 rows already absent from CSV; return {epoch -> fid} *)
  AssociationThread[rows[[All, 1]] -> rows[[All, 3]]]];

(* Propagator .dat: lines "k G sigma"; skip k = 0 row *)
loadProp[path_String] := Select[Import[path, "Table"], #[[1]] > 0 &];

gTrain = loadProp @ FileNameJoin[{corrDir, "G_k_train_celeba_128_all.dat"}];
kVals  = gTrain[[All, 1]];
gTrK   = gTrain[[All, 2]];

loadDM[method_String, epStr_String] := loadProp @ FileNameJoin[{corrDir,
   "G_k_DM_celeba_128_" <> method <> "_all_epoch=" <> epStr <> ".dat"}];

(* ============================================================ *)
(*   Aggregate metrics: D_tot, L1_pw, R_var per (method, epoch) *)
(* ============================================================ *)

Dtot[gDM_] := With[{r = gTrK / gDM[[All, 2]]},
  Total[0.5 (r - 1 - Log[r])]];

L1pw[gDM_] :=
  Total[kVals^(dim - 1) * Abs[gDM[[All, 2]] - gTrK]];

Rvar[gDM_] := Total[gDM[[All, 2]]] / Total[gTrK];

metricsFor[method_String] := Module[{fidMap = loadFid[method]},
  Table[
    With[{g = loadDM[method, epochsStr[[i]]]},
      <|"ep" -> epochs[[i]], "FID" -> fidMap[epochs[[i]]],
        "Dtot" -> Dtot[g], "L1pw" -> L1pw[g], "Rvar" -> Rvar[g]|>],
    {i, Length[epochs]}]];

emRec   = metricsFor["em"];
dpm2Rec = metricsFor["dpm2"];

(* Extract columns as plain vectors *)
ofE[rec_, key_] := rec[[All, key]];
epEm   = ofE[emRec,   "ep"];       epDp   = ofE[dpm2Rec, "ep"];
fidEm  = ofE[emRec,   "FID"];      fidDp  = ofE[dpm2Rec, "FID"];
DEm    = ofE[emRec,   "Dtot"];     DDp    = ofE[dpm2Rec, "Dtot"];
LEm    = ofE[emRec,   "L1pw"];     LDp    = ofE[dpm2Rec, "L1pw"];
REm    = ofE[emRec,   "Rvar"];     RDp    = ofE[dpm2Rec, "Rvar"];
RdevEm = Max[#, 1*^-4] & /@ Abs[REm - 1.];
RdevDp = Max[#, 1*^-4] & /@ Abs[RDp - 1.];

(* Shift epochs by +1 so ep=0 displays at x=1 on a log axis
   (matches Python plot convention) *)
xEm = epEm + 1;
xDp = epDp + 1;

(* ============================================================ *)
(*                Styles                                        *)
(* ============================================================ *)

emColor   = RGBColor[0.20, 0.43, 0.74];   (* matplotlib C0 *)
dpColor   = RGBColor[0.80, 0.22, 0.22];   (* matplotlib C3 *)
fidColor  = emColor;                      (* FID in relative-evol plots *)
DColor    = RGBColor[0.97, 0.53, 0.15];   (* orange — D_tot *)
LColor    = RGBColor[0.19, 0.63, 0.34];   (* green  — L1_pw *)
RColor    = RGBColor[0.58, 0.30, 0.63];   (* purple — |R_var-1| *)

maTeXLabel[lbl_, size_:14] := MaTeX[lbl, ContentPadding -> False, FontSize -> size];

(* ============================================================ *)
(*   Top row: relative evolution per sampler + R_var evolution  *)
(* ============================================================ *)

evolPairs[x_, F_, D_, L_, Rdev_] := {
  Transpose[{x, F / Last[F]}],
  Transpose[{x, D / Last[D]}],
  Transpose[{x, L / Last[L]}],
  Transpose[{x, Rdev / Last[Rdev]}]};

makeRelEvol[x_, F_, D_, L_, Rdev_, title_] := ListLogLogPlot[
  evolPairs[x, F, D, L, Rdev],
  Joined -> True, PlotMarkers -> Automatic,
  PlotStyle -> {fidColor, DColor, LColor, RColor},
  PlotLegends -> Placed[LineLegend[
      maTeXLabel[#, 11] & /@ {
        "\\mathrm{FID}",
        "D_{\\mathrm{tot}}",
        "L_1^{\\mathrm{pw}}",
        "|R_{\\mathrm{var}}-1|"},
      LegendMarkerSize -> {22, 6}, LegendLayout -> {"Column", 1}],
    Scaled[{0.82, 0.78}]],
  GridLines -> {None, {{1, Directive[Gray, Dashed, Opacity[0.4]]}}},
  FrameLabel -> {maTeXLabel["\\text{Epochs completed}", 12],
                 maTeXLabel["\\text{metric} / \\text{metric(ep}=9999\\text{)}", 12]},
  PlotLabel -> maTeXLabel["\\text{" <> title <> "}", 12],
  ImageSize -> 420];

(* Title strings go through \text{...} in makeRelEvol, so no math/subscripts inside *)
relEvolEm = makeRelEvol[xEm, fidEm, DEm, LEm, RdevEm, "SDE (em): relative evolution"];
relEvolDp = makeRelEvol[xDp, fidDp, DDp, LDp, RdevDp, "ODE (dpm2): relative evolution"];

rvarEvol = ListLogLinearPlot[
  {Transpose[{xEm, REm}], Transpose[{xDp, RDp}]},
  Joined -> True, PlotMarkers -> Automatic,
  PlotStyle -> {emColor, dpColor},
  PlotLegends -> Placed[LineLegend[
      maTeXLabel[#, 11] & /@ {"\\text{SDE (em)}", "\\text{ODE (dpm2)}"},
      LegendMarkerSize -> {22, 6}],
    Scaled[{0.82, 0.78}]],
  GridLines -> {None, {{1, Directive[Gray, Dashed, Opacity[0.4]]}}},
  PlotRange -> {All, {0.3, 3.0}},
  FrameLabel -> {maTeXLabel["\\text{Epochs completed}", 12],
                 maTeXLabel["R_{\\mathrm{var}} = \\sum G_{\\mathrm{DM}} / \\sum G_{\\mathrm{train}}", 12]},
  PlotLabel -> maTeXLabel["\\text{Total field-variance ratio}", 12],
  ImageSize -> 420];

(* ============================================================ *)
(*     Bottom row: FID vs each scalar (log-log scatter)         *)
(* ============================================================ *)

(* Power-law fit on log-log axes: return slope and log-intercept *)
logFit[xs_, ys_] := With[{data = Transpose[{Log@xs, Log@ys}]},
  {slope, intercept} = {b, a} /. FindFit[data, a + b x, {a, b}, x];
  {slope, intercept}];

(* `title` here is a pure MaTeX math expression (no \text{} wrapping).
   `xlabel` / `ylabel` likewise are math strings used inside the frame. *)
makeScatter[xEmV_, yEmV_, xDpV_, yDpV_, xlabel_, ylabel_, title_] :=
 Module[{fitEm, fitDp, xx, sEm, iEm, sDp, iDp, slopeStrEm, slopeStrDp},
  fitEm = logFit[xEmV, yEmV]; {sEm, iEm} = fitEm;
  fitDp = logFit[xDpV, yDpV]; {sDp, iDp} = fitDp;
  xx = Exp @ Subdivide[Log@Min[Join[xEmV, xDpV]],
                        Log@Max[Join[xEmV, xDpV]], 40];
  slopeStrEm = ToString[NumberForm[sEm, {3, 2}]];
  slopeStrDp = ToString[NumberForm[sDp, {3, 2}]];
  Show[
    ListLogLogPlot[{
        Transpose[{xEmV, yEmV}],
        Transpose[{xDpV, yDpV}]},
      PlotMarkers -> {{"\[FilledCircle]", 8}, {"\[FilledSquare]", 8}},
      PlotStyle -> {emColor, dpColor},
      PlotLegends -> Placed[LineLegend[
          maTeXLabel[#, 10] & /@ {
            "\\text{SDE (em)}",
            "\\text{ODE (dpm2)}"},
          LegendMarkerSize -> {18, 6}],
        Scaled[{0.26, 0.86}]],
      FrameLabel -> {maTeXLabel[xlabel, 12], maTeXLabel[ylabel, 12]},
      PlotLabel -> maTeXLabel[title, 12],
      ImageSize -> 420],
    LogLogPlot[{Exp[iEm]*x^sEm, Exp[iDp]*x^sDp}, {x, Min[xx], Max[xx]},
      PlotStyle -> {{emColor, Dashed, Thickness[0.004], Opacity[0.55]},
                    {dpColor, Dashed, Thickness[0.004], Opacity[0.55]}},
      PlotLegends -> Placed[LineLegend[
          {maTeXLabel["\\text{em: } y\\propto x^{" <> slopeStrEm <> "}", 10],
           maTeXLabel["\\text{dpm2: } y\\propto x^{" <> slopeStrDp <> "}", 10]},
          LegendMarkerSize -> {22, 2}],
        Scaled[{0.78, 0.22}]]]]];

panel4 = makeScatter[DEm, fidEm, DDp, fidDp,
  "D_{\\mathrm{tot}}", "\\mathrm{FID}",
  "\\mathrm{FID}\\ \\text{vs}\\ D_{\\mathrm{tot}}"];
panel5 = makeScatter[LEm, fidEm, LDp, fidDp,
  "L_1^{\\mathrm{pw}}", "\\mathrm{FID}",
  "\\mathrm{FID}\\ \\text{vs}\\ L_1^{\\mathrm{pw}}"];
panel6 = makeScatter[RdevEm, fidEm, RdevDp, fidDp,
  "|R_{\\mathrm{var}} - 1|", "\\mathrm{FID}",
  "\\mathrm{FID}\\ \\text{vs}\\ |R_{\\mathrm{var}} - 1|"];

(* ============================================================ *)
(*                       Export                                 *)
(* ============================================================ *)

Export[FileNameJoin[{figureDir, "prop_fid_em_rel_evol.pdf"}],    relEvolEm];
Export[FileNameJoin[{figureDir, "prop_fid_dpm2_rel_evol.pdf"}],  relEvolDp];
Export[FileNameJoin[{figureDir, "prop_fid_rvar_evol.pdf"}],      rvarEvol];
Export[FileNameJoin[{figureDir, "prop_fid_vs_Dtot.pdf"}],        panel4];
Export[FileNameJoin[{figureDir, "prop_fid_vs_L1pw.pdf"}],        panel5];
Export[FileNameJoin[{figureDir, "prop_fid_vs_Rvar.pdf"}],        panel6];

composite = Grid[{{relEvolEm, relEvolDp, rvarEvol},
                  {panel4,    panel5,    panel6}},
  Spacings -> {1.5, 1.5}, Frame -> None];

Export[FileNameJoin[{figureDir, "prop_fid_6panel_mma.pdf"}], composite];

Print["saved 6 individual panels + 1 composite in ", figureDir];
