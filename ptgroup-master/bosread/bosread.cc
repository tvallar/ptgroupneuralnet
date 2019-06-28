/*
Author: Paul A. Cordova, Josh Pond, Will Phelps, and jb.
Email: pac3nc@virginia.edu, jp4cj@virginia.edu/jpond@jlab.org
*/
#include <stdio.h>
#include <math.h>


/* CLAS6 headers */
#include <Vec.h>
#include <lorentz.h>
#include <pputil.h>
#include <clasEvent.h>

/* root  headers */
#include "TString.h"
#include "TFile.h"
#include "TBranch.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TSystem.h"
#include "TH2D.h"
#include "TH1F.h"
#include "TTree.h"
#include "TMatrixD.h"

#define RAD_TO_DEGREE 57.295820614
#define NDATA 52;
#define GET_VARNAME(VAR) #VAR

/* Import corrections header, modify for specific needs */
#include "/home/ptgroup/clasPack/pcor/g12pcor/All_Corrections/g12_corrections.h"

/* function prototypes */
fourVec pcorFourVec_new(fourVec particle, int part_id);
int getSector(float phi);
tagr_t *my_get_photon_tagr(clasTAGR_t *TAGR,clasBID_t *TBID, threeVec myVert);
void ProcessEvent(clasEvent &evt);

/* declare TTree which holds the data, the file which holds the data, and the
covariance matrices */
TTree *p0;
TFile *f;
Float_t pim_mat[5][5];
Float_t kp_mat[5][5];
Float_t p_mat[5][5];
clas::g12::MomentumCorrection pcor;

int main(int argc,char **argv)
{
    //variable declarations
    Int_t run, event;
    Float_t phi_pim, phi_kp, phi_p, theta_pim, theta_kp, theta_p;
    Float_t dt_pim, dt_kp, dt_p, r_pim, r_kp, r_p, t_pim, t_kp, t_p, ebeamcor;
    Float_t x_pim, x_kp, x_p, y_pim, y_kp, y_p, z_pim, z_kp, z_p;
    TLorentzVector piminus, kplus, proton;
    TMatrixD covm_kp(5,5);
    TMatrixD covm_p(5,5);
    TMatrixD covm_pim(5,5);
//    matrix<double> covm_pim;
//    matrix<double> covm_kp;
//    matrix<double> covm_p;
    //Ttree and branches
    //p0->Vector
    //TMatrix

    p0 = new TTree("p0","Data");
    p0->Branch("run",&run,"run/I");
    p0->Branch("event",&event,"event/I");
    p0->Branch("ebeamcor", &ebeamcor, "ebeamcor/F");
    p0->Branch("piminus", "TLorentzVector", &piminus, 16000, 2);
    p0->Branch("kplus", "TLorentzVector",&kplus, 16000, 2);
    p0->Branch("proton","TLorentzVector",&proton,16000, 2);
    p0->Branch("covm_pim",&covm_pim);
    p0->Branch("covm_kp",&covm_kp);
    p0->Branch("covm_p",&covm_p);

    // creates static output file named outPutFile.root, will be renamed to a unique name when Auger moves it to disk (see presentation)
    f = new TFile((char *) "outPutFile.root","recreate");
    //initiate the BOS reading
    initbos();
    for (int i = 1; i < argc; ++i) { //loop over args (input files)
        char *argptr = argv[i];
        if (*argptr != '-') { // check to make sure it's not a flag
        clasEvent evt(argptr,&bcs_,1,0); // get first event
        if (evt.status()) {
            while (evt.read(1) > 0) {//loop over events
                clasHEAD_t *HEAD;
                if (evt.type() == 1) {
                    if (HEAD = (clasHEAD_t *)getBank(&bcs_, "HEAD")) { // check for complete HEAD bank
                        event = evt.event();
                        run = evt.run();
                        //These are CLAS 4 momenta / 4 vectors for the analyzed particles. They behave in the expected way.
                        fourVec kp, p, pim;
                        //These are CLAS particle objects, they have many useful members.
                        clasParticle kpx, px, pimx;
                        fourVec beam,beam1,target;
                        double eBeamLo = 4.4;
                        double eBeamHi = 7.0;
                        double tpho; //Photon energy
                        target = evt.target().get4P();

                        clasTAGR_t *TAGR = (clasTAGR_t *)getBank(&bcs_,"TAGR"); //These are the BOS banks
                        clasBID_t *TBID  =  (clasBID_t *)getBank(&bcs_,"TBID");
                        clasTBER_t *TBER = (clasTBER_t *)getBank(&bcs_,"TBER");
                        threeVec evtVert=evt.V(); //event vertex
                        tagr_t *tagr = my_get_photon_tagr(TAGR, TBID, evtVert);
                        double beamp = -1000.0;
                        if(tagr){
                            beamp = tagr->erg;
                            tpho=tagr->tpho;
                        }
                        beam1.set(beamp,threeVec(0.0,0.0,beamp)); //set the initial tagged photon
                        if(evt.N(PiMinus)==1 && evt.N(Proton) == 1 && evt.N(KPlus) == 1){ //Part Bank PID cut
                            //Get the 4 momenta of the particles
                            pimx = evt.cp(PiMinus, 1);
                            kpx = evt.cp(KPlus, 1);
                            px = evt.cp(Proton, 1);
                            if(pimx.TBERmatrix().el(1,1) != 0 && kpx.TBERmatrix().el(1,1) != 0 && px.TBERmatrix().el(1,1) != 0){
                            evt.eLoss((char*)"g11a",1); //Event energy loss correction
                            ebeamcor = clas::g12::corrected_beam_energy(evt.run(),beam1.t()); //Beam energy correction
                            beam.set(ebeamcor,threeVec(0.0,0.0,ebeamcor)); //set corrected photon
                            // momentum corrections (takes a 4 momenta and the particle id number)
                            pim = pcorFourVec_new(pimx.p(), 9);
                            kp = pcorFourVec_new(kpx.p(), 11);
                            p = pcorFourVec_new(px.p(), 14);
                            // 4 momenta components
                            piminus.SetPxPyPzE(pim.x(), pim.y(), pim.z(), pim.t());
                            kplus.SetPxPyPzE(kp.x(), kp.y(), kp.z(), kp.t());
                            proton.SetPxPyPzE(p.x(), p.y(), p.z(), p.t());
                            // TBER matrices, used later
                            for(int j = 0; j<5; j++){
                                for(int k = 0; k<5; k++){
                                    covm_pim[j][k] = pimx.TBERmatrix().el(j,k);
                                    covm_kp[j][k] = kpx.TBERmatrix().el(j,k);
                                    covm_p[j][k] = px.TBERmatrix().el(j,k);
                                }
                            }

                            p0->Fill();// fill the TNtuple
                        }
                        }//cuts; // run analyzer
                    }
                    evt.clean();
                }
            }
        }else return 1;
    }
}
    p0->Write(); //write full TNtuple to file
    f->Close(); // close the file
    return 0;

}
//The function that returns a momentum corrected four vector, and was written by W. Phelps
fourVec pcorFourVec_new(fourVec particle, int part_id){
    float proton_mass = 0.938272; //GeV
    float pion_mass = 0.139570; //GeV
    float kaon_mass = 0.493667; //GeV
    //Assign mass based on pid.
    double PID_mass =-1000;
    PID_mass = (part_id == 14                )?proton_mass:PID_mass;
    PID_mass = (part_id == 11|| part_id == 12)?kaon_mass  :PID_mass;
    PID_mass = (part_id == 8 || part_id == 9 )?pion_mass  :PID_mass;

    float id = part_id;
    float p =  particle.r(); // total momentum
    float px =  particle.x();
    float py =  particle.y();

    float pz = sqrt(p*p - px*px - py*py);
    float phi = particle.V().phi();
    float theta = acos(pz/p);

    /// Momentum correction ////////////////////////////////////////
    float new_p = p + pcor.pcor(phi,id);
    /// ////////////////////////////////////////////////////////////

    float new_px = (new_p / p) * px;
    float new_py = (new_p / p) * py;
    float new_pz = (new_p / p) * pz;

    double pE = sqrt( PID_mass*PID_mass + new_p*new_p );
    fourVec correctedFourVec;
    correctedFourVec.set(pE, new_px, new_py, new_pz);

    return correctedFourVec;
}

//returns sector (int) of the CLAS drift chambers with inut of phi (float)
int getSector(float phi){
    int sector = 0;

    if(std::abs(phi) <= 30.) sector = 1;
    else if(phi > 0.){
        if(phi <= 90.) sector = 2;
        else if(phi <= 150) sector = 3;
        else sector = 4;
    }
    else {
        // phi < 0
        if(std::abs(phi) <= 90.) sector = 6;
        else if(std::abs(phi) <= 150.) sector = 5;
        else sector = 4;
    }
    return sector;
}

// This function returns the TAGR bank, and was written by "jb",
tagr_t *my_get_photon_tagr(clasTAGR_t *TAGR,clasBID_t *TBID, threeVec myVert){
    /* This routine only works for time-based tracks! */
    float best_diff=ST_TAG_COINCIDENCE_WINDOW;
    float tprop=0.0;
    tagr_t *tagr = NULL;
    clasTBTR_t *TBTR=(clasTBTR_t *)getBank(&bcs_,"TBTR");
    float g11targz=-10; //this was for g11
    float g12targz=-90; //jb v11
    int i, j;
    /* Exit from function if missing the requisite banks... */
    if(!TAGR || !TBTR || !TBID) return(NULL);
    for (i=0;i<TBID->bank.nrow;i++){
        int trk_ind=TBID->bid[i].track;
        if(trk_ind){
            tprop=(myVert.z() - g12targz)/LIGHT_SPEED;  //jb v13
            if (TBID->bid[i].st.stat){
                float mytime=-1000; //jb v14
                float myenergy=-1000; //jb v14
                for(j=0;j<TAGR->bank.nrow;j++){
                    float diff=fabs(TBID->bid[i].st.vtime-(TAGR->tagr[j].tpho+tprop));
                    if ((diff<ST_TAG_COINCIDENCE_WINDOW&&diff<best_diff || (abs(TAGR->tagr[j].tpho - mytime)<0.011&& (TAGR->tagr[j].erg) > myenergy)  )  &&  (TAGR->tagr[j].stat==7 || TAGR->tagr[j].stat==15)){  //jb v14
                        best_diff=diff;
                        tagr=&(TAGR->tagr[j]);
                        mytime=tagr->tpho; //jb v14
                        myenergy=tagr->erg; //jb v14
                    }
                }
            }
        }
    }
    return(tagr);
}
