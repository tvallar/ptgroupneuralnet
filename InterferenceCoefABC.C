#include <TApplication.h>
#include <TROOT.h>
#include <TMath.h>
#include <TF1.h>
#include "TGraph.h"
#include <TList.h>
#include <TLine.h>
#include <TFile.h>
#include <TCanvas.h>
#include <TString.h>
#include <THashList.h>
#include <TDirectoryFile.h>
#include <cstdlib>
#include <TTree.h>

// Global variables definition ------------------------------------

	Double_t ALP_INV = 137.0359998;
  Double_t PI = 3.1415926535;
  Double_t HBC = 0.1973269602;
  Double_t RAD = PI / 180.;

//calculation of nuclear form factors F1 and F2
//____________________________________________________________________________________
void NUCL_FF( Double_t t, Int_t charge, Double_t &F1, Double_t &F2) {

  Double_t xmas[2] = {0.939565330, 0.93827200};
  Double_t tau, shape, GE, GM0, GM;

  t = -1.*t;
  tau = 0.25 * t / xmas[charge] / xmas[charge];
  shape = 1.0 / ( 1.0 + ( t / 0.710649 ) ) / ( 1.0 + ( t / 0.710649 ) );

  if ( charge == 0 ) {

    GM0 =  -1.9130427;
    GE = 1.25 * GM0 * tau * shape / ( 1.0 + 18.3 * tau );

  } else if ( charge == 1) {

    GM0 = 2.792847337;
    GE = shape;

  }

  GM = GM0 * shape;

  F2 = ( GM - GE ) / ( 1.0 + tau );
  F1 = GM - F2;
}
//Calculation of BH cross section
//____________________________________________________________________________________
Double_t *Interf( Double_t QQ, Double_t x, Double_t t, Double_t k, Double_t phi){

	// Kinematics and 4-vectors definitions

  Double_t y, e, xi, tmin, kpr, gg, q, qp, po, pmag, cth, theta, sth, sthl, cthl, cthpr, sthpr, M, M2, TAU, X, Y, F1, F2;

  M = 0.938272;
  M2 = M*M;
  y = QQ / ( 2. * M * k * x ); // From eq. (23) where gamma is substituted from eq (12c)
	gg = 4. * M2 * x * x / QQ; // This is gamma^2 [from eq. (12c)]
	e = ( 1 - y - ( y * y * (gg / 4.) ) ) / ( 1. - y + (y * y / 2.) + ( y * y * (gg / 4.) ) ); // epsilon eq. (32)
	xi = 1. * x * ( ( 1. + t / ( 2. * QQ ) ) / ( 2. - x + x * t / QQ ) ); // skewness parameter eq. (12b) note: there is a minus sign on the write up that shouldn't be there
	tmin = ( QQ * ( 1. - sqrt( 1. + gg ) + gg / 2. ) ) / ( x * ( 1. - sqrt( 1. + gg ) + gg / ( 2.* x ) ) ); // minimum t eq. (29)
  kpr = k * ( 1. - y ); // k' from eq. (23)
  qp = t / 2. / M + k - kpr; //q' from eq. bellow to eq. (25) that has no numbering. Here nu = k - k' = k * y
  po = M - t / 2. / M; // This is p'_0 from eq. (28b)
  pmag = sqrt( ( -t ) * ( 1. - t / 4. / M / M ) ); // p' magnitude from eq. (28b)
  cth = -1. / sqrt( 1. + gg ) * ( 1. + gg / 2. * ( 1. + t / QQ ) / ( 1. + x * t / QQ ) ); // This is cos(theta) eq. (26)
	theta = acos(cth); // theta angle
  sthl = sqrt( gg ) / sqrt( 1. + gg ) * ( sqrt ( 1. - y - y * y * gg / 4. ) ); // sin(theta_l) from eq. (22a)
	cthl = -1. / sqrt( 1. + gg ) * ( 1. + y * gg / 2. ) ; // cos(theta_l) from eq. (22a)

	// 4 - momenta vectors defined on eq. (21)

	TLorentzVector K(k * sthl , 0.0,  k * cthl , k);
	TLorentzVector KP( K(0), 0.0, k * ( cthl + y * sqrt( 1. + gg ) ), kpr);
	TLorentzVector Q = K - KP;
	TLorentzVector QP(qp * sin(theta) * cos( phi * RAD ), qp * sin(theta) * sin( phi * RAD ), qp * cos(theta), qp);
  TLorentzVector P(0.0, 0.0, 0.0, M);
  TLorentzVector D = Q - QP; // delta vector eq. (12a)
  TLorentzVector PP = P + D; // p' from eq. (21)

	TLorentzVector PPK = P + K;
	// P vector eq. (12a)
	TLorentzVector PPP = P + PP;
 	PPP(0) = PPP(0) / 2.;
  PPP(1) = PPP(1) / 2.;
  PPP(2) = PPP(2) / 2.;
  PPP(3) = PPP(3) / 2.;

  Double_t D2  = D * D;
  Double_t PQS = P * Q;
  Double_t S = PPK * PPK;

  //check for comparison
  X = 0.5 * QQ / PQS;
  Y = PQS / ( P * K );

  TAU = -0.25 * D2 / M2;

  //form factors
  NUCL_FF(t,1,F1, F2);

  Double_t ESP = 2.0 * x * M / sqrt( QQ );
  Double_t ESP2 = ESP * ESP;
  Double_t ESPA = 1.0 + ESP2;
  Double_t DEPA = sqrt( ESPA );


  //kinematical factor
  Double_t HK = 18 * M2 / ( K * QP ) / ( KP * QP ) / D2 / D2;

	// 4-vectors products
	Double_t KD = K * D;
  Double_t KPD = KP * D;
	Double_t QD = Q * D;
	Double_t QPD = QP * D;
  Double_t KQP = K * QP;
	Double_t KKP = K * KP;
  Double_t KPQP = KP * QP;
	Double_t KPPP = K * PPP;
	Double_t KPPPP = KP * PPP;
	Double_t PPPQ = PPP * Q;
	Double_t PPPQP = PPP * QP;


  // Auu and Buu Twist 2 interference coefficients
	Double_t KK_T, KQP_T, KKP_T, KXQP_T, KD_T, DD_T, Auu, Buu, Cuu, Aul;

	// Transverse components defined after eq.(241c) -----------------------
	KK_T = 0.5 * ( e / ( 1 - e ) ) * QQ;

	KKP_T = KK_T;

	KQP_T = ( QQ / ( sqrt( gg ) * sqrt( 1 + gg ) ) ) * sqrt ( (0.5 * e) / ( 1 - e ) ) * ( 1. + x * t / QQ ) * sin(theta) * cos( phi * RAD );

	KD_T = -1.* KQP_T;

	DD_T = ( 1. - xi * xi ) * ( tmin - t );

	// Interference coefficients given on eq. (241a,b,c)--------------------
	Auu = -4.0 * cos( phi * RAD ) / (KQP * KPQP) * ( ( QQ + t ) * ( 2.0 * ( KPPP + KPPPP ) * KK_T   + ( PPPQ * KQP_T ) + 2.* ( KPPPP * KQP ) - 2.* ( KPPP * KPQP ) ) +
				 										    ( QQ - t + 4.* KD ) * PPPQP * ( KKP_T + KQP_T - 2.* KKP ) );

	Buu = 2.0 * xi * cos( phi * RAD ) / ( KQP * KPQP) * ( ( QQ + t ) * ( 2.* KK_T * ( KD + KPD ) + KQP_T * ( QD - KQP - KPQP + 2.*KKP ) + 2.* KQP * KPD - 2.* KPQP * KD ) +
																		       ( QQ - t + 4.* KD ) * ( ( KK_T - 2.* KKP ) * QPD - KKP * DD_T - 2.* KD_T * KQP ) ) / TAU;

	Cuu = 2.0 * cos( phi * RAD ) / ( KQP * KPQP) * ( -1. * ( QQ + t ) * ( 2.* KKP - KPQP - KQP ) * KD_T + ( QQ - t + 4.* KD ) * ( ( KQP + KPQP ) * KD_T + DD_T * KKP ) );

	//Cuu = 2.0 * cos( phi * RAD ) / ( KQP * KPQP) * ( -1. * ( QQ + t ) * ( 2.* KKP - KPQP - KQP ) * KQP_T * (sqrt( 1 + gg )) + ( QQ - t + 4.* KD ) * ( - ( KQP + KPQP ) * KQP_T * sqrt( 1 + gg ) + DD_T * KKP ) ); //  From Simonetta code
	// ---------------------------------------------------------------------

	Double_t GAMMA = 1. / ALP_INV / ALP_INV / ALP_INV / PI / PI / 16. / ( S - M2 ) / ( S - M2 ) / sqrt( 1. + gg ) / x ; // eq.(10)


	// Axial Form Factor Ga - From Simonetta's code (used to find the interference cross section with the Cuu coeffient term i.e. XSINTC)
	Double_t a = 1.267 / ( pow( 1. - t / pow(1.026, 2) , 2) );
	Double_t b = 0.585 / ( pow( 1. - t / pow(1.2, 2) , 2) );

	Double_t Ga = ( a + 3. * b ) * ( a + 3. * b ) / 4.;

	// Interference cross section
	Double_t XSINT =  GAMMA / (QQ * -1.* t) * ( Auu * ( F1 * F1 + TAU * F2 * F2 ) + Buu * TAU * ( ( F1 + F2 ) * ( F1 + F2 ) ) ); // Cuu assumed to be negligible
	Double_t XSINTC =  GAMMA / (QQ * -1.* t) * ( Auu * ( F1 * F1 + TAU * F2 * F2 ) + Buu * TAU * ( ( F1 + F2 ) * ( F1 + F2 ) ) + Cuu * ( ( F1 + F2 ) * Ga ) ); // with Cuu term

	// Normalization --------------------------------------------------------------
	// Use Defurne's Jacobian
  Double_t conv1 = 1./ ( 2. * M * x * K(3) ) * 2. * PI * 2.;

	//from dy to dQ^2
	Double_t phas_1 = 1.0;
	Double_t phas_2 = 1.0;

	XSINT  *= 10000000. * HBC * HBC * phas_1 * phas_2 * conv1;
	XSINTC *= 10000000. * HBC * HBC * phas_1 * phas_2 * conv1;

	Double_t AuuTerm = Auu * ( F1 * F1 + TAU * F2 * F2 );
	Double_t BuuTerm = Buu * TAU * ( ( F1 + F2 ) * ( F1 + F2 ) ) ;
	Double_t CuuTerm = Cuu * ( ( F1 + F2 ) * Ga );

	AuuTerm *= 10000000. * HBC * HBC * phas_1 * phas_2 * conv1 * GAMMA / (QQ * -1.* t);
	BuuTerm *= 10000000. * HBC * HBC * phas_1 * phas_2 * conv1 * GAMMA / (QQ * -1.* t);
	CuuTerm *= 10000000. * HBC * HBC * phas_1 * phas_2 * conv1 * GAMMA / (QQ * -1.* t);

  //Double_t rXSINT = (Auu + Buu) / Buu;

	Double_t AuuTauOverBuu = Auu / Buu;
	//Double_t AuuTauOverBuu = AuuTerm / BuuTerm;

	Double_t rXSOverBuu =  AuuTauOverBuu * ( F1 * F1 + TAU * F2 * F2 ) + TAU * ( ( F1 + F2 ) * ( F1 + F2 ) ) ;
	//Double_t rXSOverBuu = XSINT * (QQ * -1.* t) / Buu / GAMMA ;

  Double_t *Output = new Double_t [10];

	Output[0] = XSINT;
	Output[1] = XSINTC;
  Output[2] = AuuTerm;
  Output[3] = BuuTerm;
	Output[4] = CuuTerm;
	Output[5] = Auu/cos( phi * RAD );
	Output[6] = Buu * TAU /cos( phi * RAD );
	Output[7] = Cuu/cos( phi * RAD );
	Output[8] = AuuTauOverBuu;
	Output[9] = rXSOverBuu;

  return Output;
}

//____________________________________________________________________________________
//void Interference2( Double_t QQ, Double_t x, Double_t t, Double_t k ) {
void InterferenceCoefABC() {
	const Int_t nval = 6;
	Double_t k = 5.75; //GeV
	Double_t t[nval] = {-0.152, -0.172, -0.192, -0.25, -0.245, -0.248};
	Double_t x = 0.348; //xB
	Double_t QQ = 1.82;

	TFile* f = TFile::Open("./Interference.root", "RECREATE");
	TTree* fTree[nval];
	Double_t phi, xsIntf, xsIntfC, AuuTerm, BuuTerm, CuuTerm, Auu, Buu, Cuu, AuuTauOverBuu, rXSOverBuu, cosphi;
	for (Int_t i = 0; i < nval; i++)	{ // Sensor loop
		fTree[i]	= new TTree(Form("Intf_xB_%.3f_QQ_%.2f_k_%.2f_t_%.3f", x, QQ, k, t[i]),"Interference calculation variables");
		fTree[i] ->Branch("phi",&phi,"phi/D");
		fTree[i] ->Branch("cosphi",&cosphi,"cosphi/D");
		fTree[i] ->Branch("xsIntf",&xsIntf,"xsIntf/D");
		fTree[i] ->Branch("xsIntfC",&xsIntfC,"xsIntfC/D");
		fTree[i] ->Branch("AuuTerm",&AuuTerm,"AuuTerm/D");
		fTree[i] ->Branch("BuuTerm",&BuuTerm,"BuuTerm/D");
		fTree[i] ->Branch("CuuTerm",&CuuTerm,"CuuTerm/D");
		fTree[i] ->Branch("Auu",&Auu,"Auu/D");
		fTree[i] ->Branch("Buu",&Buu,"Buu/D");
		fTree[i] ->Branch("Cuu",&Cuu,"Cuu/D");
		fTree[i] ->Branch("AuuTauOverBuu",&AuuTauOverBuu,"AuuTauOverBuu/D");
		fTree[i] ->Branch("rXSOverBuu",&rXSOverBuu,"rXSOverBuu/D");
	}
	Double_t phiLow = 0.0;
	Int_t npoints = 361;

	//  Fill the tree
	for(Int_t ikin = 0; ikin < nval; ikin++) {
		for(Int_t i = 0; i < npoints; i+=1) {

			phi = phiLow + (Double_t)i;
			Double_t *GetOutput = Interf( QQ, x, t[ikin], k, phi);
			xsIntf = GetOutput[0];
			xsIntfC = GetOutput[1];
			AuuTerm = GetOutput[2];
			BuuTerm = GetOutput[3];
			CuuTerm = GetOutput[4];
			Auu = GetOutput[5];
			Buu = GetOutput[6];
			Cuu = GetOutput[7];
			AuuTauOverBuu = GetOutput[8];
			rXSOverBuu = GetOutput[9];

			cosphi = cos( phi * RAD );

			fTree[ikin] ->Fill();


		}
	}


	TH2F *htemp;

	// Draw 2dim hist from tree --------------------------------------------------------------------------
	TCanvas *MyC_B = new TCanvas("MyC_B","Test canvas",1);
	MyC_B->Divide(2,3);

	MyC_B->cd(1);
	fTree[1] ->Draw("Cuu:phi","","L");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[1]));
	gPad ->Update();

	MyC_B->cd(2);
	fTree[3] ->Draw("Buu:phi","","P");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[3]));
	gPad ->Update();

	MyC_B->cd(3);
	fTree[4] ->Draw("Cuu:phi","","L");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[4]));
	gPad ->Update();

	MyC_B->cd(4);
	fTree[4] ->Draw("Buu:phi","","L");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[4]));
	gPad ->Update();

	MyC_B->cd(5);
	fTree[5] ->Draw("Cuu:phi","","L");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[5]));
	gPad ->Update();

	MyC_B->cd(6);
	fTree[5] ->Draw("Buu:phi","","L");
	htemp = (TH2F*)gPad->GetPrimitive("htemp");
	htemp->SetTitle(Form("k = %.2f GeV,Q^{2} = %.2f GeV^{2},x_{B} = %.3f,t = %.3f", k, QQ, x, t[5]));
	gPad ->Update();


	Int_t n1 = fTree[1]->Draw("AuuTerm:phi","","goff");
	TGraph *gAuuTerm = new TGraph(n1,fTree[1]->GetV2(),fTree[1]->GetV1());
				gAuuTerm ->SetLineColor(2);

	Int_t n2 = fTree[1]->Draw("BuuTerm:phi","","goff");
	TGraph *gBuuTerm = new TGraph(n2,fTree[1]->GetV2(),fTree[1]->GetV1());
				gBuuTerm ->SetLineColor(4);

	Int_t n3 = fTree[1]->Draw("CuuTerm:phi","","goff");
	TGraph *gCuuTerm = new TGraph(n3,fTree[1]->GetV2(),fTree[1]->GetV1());
				gCuuTerm ->SetLineColor(6);

	Int_t n4 = fTree[1]->Draw("xsIntfC:phi","","goff");
	TGraph *gXSIntC = new TGraph(n4,fTree[1]->GetV2(),fTree[1]->GetV1());
				gXSIntC ->SetLineColor(1);


 	TMultiGraph *mgr = new TMultiGraph();
 		mgr ->SetTitle(";#phi [#circ];d^{4}#sigma [10^{-3}nb/GeV^{4}]");
 		mgr->Add(gAuuTerm);
 		mgr->Add(gBuuTerm);
		mgr->Add(gCuuTerm);
 		mgr->Add(gXSIntC);

// // ---------- Draw Graphs -------------------------------------------------
	//gROOT->Macro("style2.C");

	TText *pt = new TText(0.5,0.6,"");

	TLegend pLegend(0.1,0.9,0.91,0.94);
		pLegend.SetTextSize(0.022);
		pLegend.AddEntry(pt, Form("k = %.2f GeV, Q^{2} = %.2f GeV^{2}, x_{B} = %.3f, t = %.3f", k, QQ, x, t[1]),"");
		pLegend.SetFillColor(10);
		pLegend.SetBorderSize(1);

	TLegend pLegend2(0.75,0.7,0.85,0.85);
		pLegend2.SetTextSize(0.02);
		pLegend2.AddEntry(gXSIntC," #sigma^{I}","L");
		pLegend2.AddEntry(gAuuTerm," A^{I}_{UU}","L");
		pLegend2.AddEntry(gBuuTerm," B^{I}_{UU}","L");
		pLegend2.AddEntry(gCuuTerm," C^{I}_{UU}","L");
		pLegend2.SetFillColor(10);
		pLegend2.SetBorderSize(0);

	TString * psfileName3 = new TString("./xsVsphiC.eps");
		gStyle->SetPaperSize(20,20);
		TCanvas* c3 = new TCanvas("c3","IM3",0,0,400,400);
		c3->SetLeftMargin(0.1);
		c3->SetRightMargin(0.09);
		TPostScript * ps3 = new TPostScript(psfileName3->Data(),-113);
		ps3->Range(20,20);
		//TGaxis::SetMaxDigits(3);
		//gPad->SetLogy();

 		mgr ->Draw("AL");

		TAxis *xaxis3 = mgr->GetXaxis();
		TAxis *yaxis3 = mgr->GetYaxis();

		xaxis3 ->SetLabelSize(0.023);
		yaxis3 ->SetLabelSize(0.023);
		xaxis3 ->SetTitleOffset(2);
		yaxis3 ->SetTitleOffset(2.1);
		xaxis3 ->SetTitleSize(0.021);
		yaxis3 ->SetTitleSize(0.021);

 		pLegend.Draw();
 		pLegend2.Draw();
 		ps3->Close();



	delete f;
}