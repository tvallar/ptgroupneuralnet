//_____________________________________________________________________________
// Standard Headers:
#include <fstream>
#include <cmath>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "TFile.h"
#include "TTree.h"
#include "TH1.h"
#include "TH2F.h"
#include "TF1.h"
#include "TMath.h"
#include "TString.h"
#include "TSystem.h"
#include "TFile.h"
#include "TROOT.h"
#include <TVector3.h>
#include "TLorentzVector.h"
#include <string.h>
#include "TApplication.h"
#include "kinline.h"
#include "kinline.cc"
#include "kstream.h"
#include "kstream.cc"
#include "TMatrixD.h"

#define NUM_PART 3
#define NUM_PART_FIN 4
// 0 for gam, 1 for pi0
#define SASQUATCH 1

using namespace std;

TMatrixD CombineMatrices(std::vector<TMatrixD> input_matrices);


int main(int __argc,char *__argv[]){

	Int_t Argc = __argc;
	char **Input = __argv;
	std::vector<string> w;
	w.assign(__argv,__argv + __argc);
	TApplication*  theApp = new TApplication("App",&__argc,__argv);
	char *outFileName = "mergeplot.root";
	extern int optind;
	TFile outFile(outFileName,"recreate");
	TTree *b;
	//gROOT->Reset();
	//TROOT troot();
	Int_t Sig;
	Float_t pull0, pull1, pull2, pull3, pull4, pull5, pull6, pull7, pull8, pull9, chisq, prob;
	Float_t pi_pull0, pi_pull1, pi_pull2, pi_pull3, pi_pull4, pi_pull5, pi_pull6, pi_pull7, pi_pull8, pi_pull9, pi_chisq,pi_prob;
	Float_t x_pim, kp_theta, mm2, kp_phi, kps_m, kp_me, kp_mp, me, mp, mtheta, mphi;
	TLorentzVector *piminus=0;
	TLorentzVector *kplus=0;
	TLorentzVector *proton=0;
	TLorentzVector P4pho, P4target, total, missing, lambda, kplus_star, sigmazero8;

	TVector3 *Vpim = new TVector3;
	TVector3 *Vkp = new TVector3;
	TVector3 *Vp = new TVector3;

    TMatrixD *obo = new TMatrixD(1,1);
	TMatrixD *covm_pim = new TMatrixD;
	TMatrixD *covm_kp = new TMatrixD;
	TMatrixD *covm_p = new TMatrixD;
	// TMatrixD.Print()
	
	TH1F *sigmazero8_mm = new TH1F("MMsigmazero8","test",200,0,3);
        TH1F *lambda_mm = new TH1F("MMlambda","test",200,0,2);
        TH1F *kplus_star_mm = new TH1F("MMKplusstar","test",200,0,2);
        TH1F *missing_mm2 = new TH1F("MM2missing","test",200,-0.2,0.5);
        TH1F *missing_mm = new TH1F("MMmissing","test",200,-0.2,0.5);
        TH1F *gamma_prob = new TH1F("gamma_prob","test",200,0,1);
        TH1F *gamma_chisq = new TH1F("gamma_chisq","test",200,0,20);

	Float_t sigmazero_m;

	Float_t ebeamcor;

	Int_t Pi_count = 0;
	Int_t G_count = 0;

	b = new TTree("background","data");
	b->Branch("pull0",&pull0,"pull0/F");
	b->Branch("pull1",&pull1,"pull1/F");
	b->Branch("pull2",&pull2,"pull2/F");
	b->Branch("pull3",&pull3,"pull3/F");
	b->Branch("pull4",&pull4,"pull4/F");
	b->Branch("pull5",&pull5,"pull5/F");
	b->Branch("pull6",&pull6,"pull6/F");
	b->Branch("pull7",&pull7,"pull7/F");
	b->Branch("pull8",&pull8,"pull8/F");
	b->Branch("pull9",&pull9,"pull9/F");
	b->Branch("pi_pull0",&pi_pull0,"pi_pull0/F");
	b->Branch("pi_pull1",&pi_pull1,"pi_pull1/F");
	b->Branch("pi_pull2",&pi_pull2,"pi_pull2/F");
	b->Branch("pi_pull3",&pi_pull3,"pi_pull3/F");
	b->Branch("pi_pull4",&pi_pull4,"pi_pull4/F");
	b->Branch("pi_pull5",&pi_pull5,"pi_pull5/F");
	b->Branch("pi_pull6",&pi_pull6,"pi_pull6/F");
	b->Branch("pi_pull7",&pi_pull7,"pi_pull7/F");
	b->Branch("pi_pull8",&pi_pull8,"pi_pull8/F");
	b->Branch("pi_pull9",&pi_pull9,"pi_pull9/F");
	b->Branch("chisq",&chisq,"chisq/F");
	b->Branch("prob",&prob,"prob/F");
	b->Branch("pi_chisq",&pi_chisq,"pi_chisq/F");
	b->Branch("pi_prob",&pi_prob,"pi_prob/F");
	b->Branch("mm2", &mm2, "mm2/F");
	b->Branch("me", &me, "me/F");
	b->Branch("mp", &mp, "mp/F");
	b->Branch("mtheta", &mtheta, "mtheta/F");
	b->Branch("mphi", &mphi, "mphi/F");
	b->Branch("kps_m", &kps_m, "kps_m/F");
	b->Branch("kp_me", &kp_me, "kp_me/F");
	b->Branch("kp_mp", &kp_mp, "kp_mp/F");
	b->Branch("kp_theta", &kp_theta, "kp_theta/F");
	b->Branch("kp_phi", &kp_phi, "kp_phi/F");
	b->Branch("sigmazero_m",&sigmazero_m, "sigmazero_m/F");

	b->Branch("Sig",&Sig,"Sig/I");

	for(int n_arg = optind; n_arg < Argc; n_arg++){
		TString input = w[n_arg];
		TFile inFile(input); // open the input file

		if(TTree *p0 = (TTree*)inFile.Get("p0")){
			p0->SetBranchAddress("piminus",&piminus);
			p0->SetBranchAddress("kplus", &kplus);
			p0->SetBranchAddress("proton",&proton);
			p0->SetBranchAddress("ebeamcor", &ebeamcor);
			p0->SetBranchAddress("covm_pim",&covm_pim);
			p0->SetBranchAddress("covm_kp",&covm_kp);
			p0->SetBranchAddress("covm_p",&covm_p);


			Int_t nentries = (Int_t)p0->GetEntries();

			for (Int_t j=0;j<=nentries;j++) {

				Double_t m_targ = 0.93828;
				std::string exp = "g12";
				p0->GetEntry(j);
				P4target.SetPxPyPzE(0.0,0.0,0.0,0.93828);
				P4pho.SetPxPyPzE(0.0,0.0,ebeamcor,ebeamcor);

                obo->Zero();
				covm_pim->ResizeTo(3,3);
				covm_kp->ResizeTo(3,3);
				covm_p->ResizeTo(3,3);


				vector <TMatrixD> vec_covm;
                vec_covm.push_back(*obo);
				vec_covm.push_back(*covm_pim);
				vec_covm.push_back(*covm_kp);
				vec_covm.push_back(*covm_p);

				TMatrixD covm_tmp = CombineMatrices(vec_covm);

				//covm_tmp.ResizeTo(10,10);

				std::vector<string> particles = {"pim","kp","p"};
				std::vector<TLorentzVector> p4 = {*piminus,*kplus,*proton};
				std::vector<TVector3> vert = {*Vpim,*Vkp,*Vp};

				TMatrixD covm = CorrectCLAS_V(covm_tmp,particles,p4,vert,true,true,exp);

				Kstream gamma;
				gamma.StringNames(particles);
				gamma.FitInput(ebeamcor,p4,covm,m_targ);

				gamma.Fit("gamma");
				chisq = gamma.Chi2();
				prob = gamma.Prob();

				Kstream pi0;
				pi0.StringNames(particles);
				pi0.FitInput(ebeamcor,p4,covm,m_targ);

				pi0.Fit("pi0");
				pi_chisq = pi0.Chi2();
				pi_prob = pi0.Prob();

				pull0 = gamma.GetPull(0);
				pull1 = gamma.GetPull(1);
				pull2 = gamma.GetPull(2);
				pull3 = gamma.GetPull(3);
				pull4 = gamma.GetPull(4);
				pull5 = gamma.GetPull(5);
				pull6 = gamma.GetPull(6);
				pull7 = gamma.GetPull(7);
				pull8 = gamma.GetPull(8);
				pull9 = gamma.GetPull(9);
				pi_pull0 = pi0.GetPull(0);
				pi_pull1 = pi0.GetPull(1);
				pi_pull2 = pi0.GetPull(2);
				pi_pull3 = pi0.GetPull(3);
				pi_pull4 = pi0.GetPull(4);
				pi_pull5 = pi0.GetPull(5);
				pi_pull6 = pi0.GetPull(6);
				pi_pull7 = pi0.GetPull(7);
				pi_pull8 = pi0.GetPull(8);
				pi_pull9 = pi0.GetPull(9);

				lambda = *proton + *piminus;
				total = P4target + P4pho;
				kplus_star = total - lambda;
				missing = total - *kplus - lambda;
				sigmazero8 = lambda + missing;

				mm2 = missing.M2();
				me = missing.E();
				mp = missing.P();
				mtheta = missing.Theta();
				mphi = missing.Phi();
				kps_m = kplus_star.M();
				kp_me = kplus->E();
				kp_mp = kplus->P();
				kp_phi = kplus->Phi();
				kp_theta = kplus->Theta();
				sigmazero_m = sigmazero8.M();
				Sig = SASQUATCH;

                                lambda_mm->Fill(lambda.M());
                                kplus_star_mm->Fill(kplus_star.M());
                                missing_mm->Fill(missing.M());
                                missing_mm2->Fill(missing.M2());
				sigmazero8_mm->Fill(sigmazero8.M());

				b->Fill();

			}
		}
	}
	outFile.Write(); // write to the output file
	outFile.Close(); // close the output file
}//end of main

TMatrixD CombineMatrices(std::vector<TMatrixD> input_matrices){

	// Get the size of the combined matrix:
	int total_rows = 0;
	for(int m = 0; m < (int) input_matrices.size(); m++){
		total_rows += input_matrices[m].GetNrows();
	}

	// Combine the input matrices together
	TMatrixD combined_matrix(total_rows, total_rows);
	int sum_nrows = 0;
	for(int m = 0; m < (int) input_matrices.size(); m++){
		int nrows = input_matrices[m].GetNrows();
		int lower_bound = sum_nrows;
		int upper_bound = sum_nrows + nrows - 1;
		TMatrixDSub(combined_matrix, lower_bound, upper_bound, lower_bound, upper_bound) = input_matrices[m];
		sum_nrows += nrows;
	}

	return combined_matrix;
}
