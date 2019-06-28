#include "KinLine.h"
#include <cmath>
#include "TDecompQRH.h"
#include "MultiScat.c"
#include <string>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <iostream>
#include <locale>
#include <string>

using std::cout;
using std::ctype;
using std::endl;
using std::locale;
using std::string;
using std::transform;
using std::use_facet;

 double AlphaTrack(const TLorentzVector &__p4){
   
   int sector = 0;
   double pi = 3.14159;
   double phi_lab = __p4.Phi();
   double phi = (180./pi)*phi_lab;
   
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
   return (pi/3.)*(sector - 1);
 }
//_____________________________________________________________________________

/// Calculates the tracking angle \f$ \lambda \f$.
double LambdaTrack(const TLorentzVector &__p4){
  
  double lambda;
  double p_mag = __p4.P();
  double x = __p4.X()/p_mag,y = __p4.Y()/p_mag;
  
  double alpha = AlphaTrack(__p4);
  lambda = asin(cos(alpha)*y - sin(alpha)*x);
  return lambda;
}
//_____________________________________________________________________________

/// Calculates the tracking angle \f$ \phi \f$.
double PhiTrack(const TLorentzVector &__p4) {
  
  double phi;
  double z = __p4.Pz()/__p4.P(); // normalized z_lab
  double lambda = LambdaTrack(__p4);
  
  phi = acos(z/cos(lambda));
  return phi;
}

//To correct CLAS tracking matrix
/* 
   C00 = pow(0.001*eBeamEnergy=4.018),2)/3;
   for each charged particle
   C11 = c11*pow(p_mag,4);
   C22 = c22;
   C33 = c33;
   C12 = -c12*q*p_mag*p_mag;
   C21 = C12;
   C13 = -c13*q*p_mag*p_mag;
   C31 = C13;
   C23 = c23;
   C32 = C23; 
*/

TMatrixD CorrectCLAS_V(const TMatrixD &__VTrack,
		       const std::vector<String> &__p_names,
                       const std::vector<TLorentzVector> &__p4,
                       const std::vector<TVector3> &__vert,
                       bool __multi,bool __isMC,String &__xid){
  
   TMatrixD V(__VTrack); // start it off as the tracking Variance matrix
   
   // VTrack must be square
   if(__VTrack.GetNrows() != __VTrack.GetNcols()){
     std::cout << "Error! <CorrectCLAS_V> Tracking Variance matrix passed to "
               << "this function is NOT square. Size is " 
               << __VTrack.GetNrows() << " x " << __VTrack.GetNcols()
               << std::endl;
     abort();
   }
 
   int numParts = (int)__p4.size();  
   if(__VTrack.GetNrows() != (3*numParts + 1)){
     std::cout << "Error! <CorrectCLAS_V> Tracking Variance matrix size is NOT"
               << " consistent with number of particle 4-momenta passed in. "
               << numParts << " particles given so Variance matrix should be "
               << (3*numParts+1) << " x " << (3*numParts+1) << " but has "
               << __VTrack.GetNrows() << " rows and columns." << std::endl;
     abort();
   }
   
   static bool first_call = true;
   static double pPars[2][6][15][3],lamPars[2][6][15],phiPars[2][6][15];
   if(first_call) {
     // initialize the parameter arrays
     ReadInResParams(pPars,lamPars,phiPars,__xid);
     first_call = false;
   }
   
   double res_scale=1,p_res_scale=1,p_scale=1,r_scale=1,sigma_eloss,lam_sig_ms,phi_sig_ms,p_offset;
   int part_index,sector,bin,cell=7;
   if(__xid != "g10a" && __xid != "g11a" && __xid != "g12" && __xid != "g13a"){ 
     std::cout<<"Experiment not known, use either g10a, g11a, g12 or g13"<<std::endl;
     abort();
   }
   
   double e_gamma_res;
   if(__xid == "g10a") e_gamma_res = 3.776;   
   if(__xid == "g11a") e_gamma_res = 4.018;
   if(__xid == "g12") e_gamma_res = 5.715;
   if(__xid == "g13a") e_gamma_res = 2.655;
   
   e_gamma_res *= 0.001/sqrt(3.);   
   
   if(__xid == "g10a"){ res_scale = 1.2;
     cell =6;
     if(__isMC) res_scale *= 1.3;
   }     
   if(__xid == "g11a"){ res_scale = 1.5;
     p_scale = 1.0;
     r_scale = 1.0;
     cell = 7;
     if(__isMC) res_scale *= 1.4;
   }  
   if(__xid == "g12"){ res_scale = 2.83;
     p_scale = 1.94;
     r_scale = 1.3;     
     cell = 7;
     if(__isMC) res_scale *= 1.1;
   }   
   if(__xid == "g13a"){ res_scale = 4.9;
     p_scale = 2.4;
     r_scale = 1.7;   
     cell =6;   
     if(__isMC) res_scale *= 1.12;
   }   

   double ecP_sig, ecTh_sig, ecPhi_sig;
   // set up the Variance matrix
   V(0,0) = pow(e_gamma_res,2); // tagged photon sigma^2
   float ms,el;
   double beta,theta,thetad,phid,phi2,p_mag,sthetsq,gamma;

   for(int i = 0; i < numParts; i++){      
     beta = __p4[i].Beta();
     gamma = __p4[i].Gamma();
     theta = __p4[i].Theta();
     p_mag = __p4[i].P();
     
     thetad = theta * 180./3.1416;
     phid = __p4[i].Phi() * 180./3.1416;
     
     //convert phi to EC coordinates
     if(phid < -30.) phi2 = phid + 360.;
     else phi2 = phid;
     phid = fmod(phi2+30.,60.) - 30.;
     
     if (__p_names[i] == "n") { 		 			 		
        ecP_sig = 3*(0.02706-0.05436*p_mag+0.0737*p_mag*p_mag);
	ecPhi_sig = 3*(5.808-139.8*pow(theta,1)+1463*pow(theta,2)-8634*pow(theta,3)+31700*pow(theta,4)-75220*pow(theta,5)+115600*pow(theta,6)-111300*pow(theta,7)+61010*pow(theta,8)-14530*pow(theta,9));
	ecTh_sig =2*(0.01885-0.008987*pow(p_mag,1)+0.003421*pow(p_mag,2));
	V(3*i+1,3*i+1) = pow(ecP_sig,2);
	V(3*i+1,3*i+2) = 0.25*ecTh_sig*ecP_sig;//V(1,2) is p theta correlation
	V(3*i+1,3*i+3) = 0.0;//V(1,3) is p phi correlation
	V(3*i+2,3*i+1) = 0.25*ecTh_sig*ecP_sig;//V(2,1) is p theta correlation 
	V(3*i+2,3*i+2) = pow(ecTh_sig,2);
	V(3*i+2,3*i+3) = 0.0;//V(2,3) is theta phi correlation                      
	V(3*i+3,3*i+1) = 0.0;//V(3,1) is p phi correlation
	V(3*i+3,3*i+2) = 0.0;//V(3,2) is theta phi correlation
	V(3*i+3,3*i+3) = pow(ecPhi_sig,2);       
     }
     if (__p_names[i] == "gamma" && __xid == "g11a"){	
       ecP_sig = 0.0945*sqrt(p_mag);
       if(__isMC) ecP_sig = 0.093*sqrt(p_mag)+0.003;
       ecTh_sig = (0.04552-0.1343*p_mag+0.2281*p_mag*p_mag-0.1631*p_mag*p_mag*p_mag+0.04339*p_mag*p_mag*p_mag*p_mag);
       ecPhi_sig =0.9*(0.1414-0.3961*p_mag+0.427*p_mag*p_mag-0.04099*p_mag*p_mag*p_mag-0.1203*p_mag*p_mag*p_mag*p_mag);      
       V(3*i+1,3*i+1) = pow(ecP_sig,2);
       V(3*i+1,3*i+2) = 0.4*ecTh_sig*ecP_sig;
       V(3*i+1,3*i+3) = 0.01*ecP_sig*ecPhi_sig;
       V(3*i+2,3*i+1) = 0.4*ecTh_sig*ecP_sig;  
       V(3*i+2,3*i+2) = pow(ecTh_sig,2);
       V(3*i+2,3*i+3) = 0.008*ecTh_sig*ecPhi_sig;                        
       V(3*i+3,3*i+1) = 0.01*ecP_sig*ecPhi_sig;
       V(3*i+3,3*i+2) = 0.008*ecTh_sig*ecPhi_sig;
       V(3*i+3,3*i+3) = pow(ecPhi_sig,2);   
     }     
    if (__p_names[i] == "gamma" && __xid == "g12"){	
       ecP_sig = 0.106*sqrt(p_mag);
       if(__isMC) ecP_sig = 0.093*sqrt(p_mag)+0.003;
       ecTh_sig = (0.04552-0.1343*p_mag+0.2281*p_mag*p_mag-0.1631*p_mag*p_mag*p_mag+0.04339*p_mag*p_mag*p_mag*p_mag);
       ecPhi_sig =0.7*(0.1414-0.3961*p_mag+0.427*p_mag*p_mag-0.04099*p_mag*p_mag*p_mag-0.1203*p_mag*p_mag*p_mag*p_mag);      
       V(3*i+1,3*i+1) = pow(ecP_sig,2);
       V(3*i+1,3*i+2) = 0.38*ecTh_sig*ecP_sig;
       V(3*i+1,3*i+3) = 0.011*ecP_sig*ecPhi_sig;
       V(3*i+2,3*i+1) = 0.41*ecTh_sig*ecP_sig;  
       V(3*i+2,3*i+2) = pow(ecTh_sig,2);
       V(3*i+2,3*i+3) = 0.0078*ecTh_sig*ecPhi_sig;                        
       V(3*i+3,3*i+1) = 0.012*ecP_sig*ecPhi_sig;
       V(3*i+3,3*i+2) = 0.0082*ecTh_sig*ecPhi_sig;
       V(3*i+3,3*i+3) = pow(ecPhi_sig,2);   
      }     
    if (__p_names[i] == "gamma" && __xid == "g13a"){	
       ecP_sig = 0.90*(0.109-0.3739*p_mag+0.6731*p_mag*p_mag);                     
       if(__isMC) ecP_sig = 0.093*sqrt(p_mag)+0.003;
       ecTh_sig = (0.04552-0.1343*p_mag+0.2281*p_mag*p_mag-0.1631*p_mag*p_mag*p_mag+0.04339*p_mag*p_mag*p_mag*p_mag);
       ecPhi_sig =(0.1414-0.3961*p_mag+0.427*p_mag*p_mag-0.04099*p_mag*p_mag*p_mag-0.1203*p_mag*p_mag*p_mag*p_mag);      
       V(3*i+1,3*i+1) = pow(ecP_sig,2);
       V(3*i+1,3*i+2) = 0.39*ecTh_sig*ecP_sig;
       V(3*i+1,3*i+3) = 0.011*ecP_sig*ecPhi_sig;
       V(3*i+2,3*i+1) = 0.385*ecTh_sig*ecP_sig;  
       V(3*i+2,3*i+2) = pow(ecTh_sig,2);
       V(3*i+2,3*i+3) = 0.0078*ecTh_sig*ecPhi_sig;                        
       V(3*i+3,3*i+1) = 0.012*ecP_sig*ecPhi_sig;
       V(3*i+3,3*i+2) = 0.0076*ecTh_sig*ecPhi_sig;
       V(3*i+3,3*i+3) = pow(ecPhi_sig,2);   
     }
    if (__p_names[i] == "gamma" && __xid == "g10a"){	
       ecP_sig = 0.096*sqrt(p_mag);
       if(__isMC) ecP_sig = 0.093*sqrt(p_mag)+0.003;
       ecTh_sig = pow(0.023/5.45,2);
       ecPhi_sig = pow(0.023/(5.45*sin(thetad)),2);
       V(3*i+1,3*i+1) = pow(ecP_sig,2);
       V(3*i+1,3*i+2) = 0.3*ecTh_sig*ecP_sig;
       V(3*i+1,3*i+3) = 0.02*ecP_sig*ecPhi_sig;
       V(3*i+2,3*i+1) = 0.33*ecTh_sig*ecP_sig;  
       V(3*i+2,3*i+2) = pow(ecTh_sig,2);
       V(3*i+2,3*i+3) = 0.007*ecTh_sig*ecPhi_sig;                        
       V(3*i+3,3*i+1) = 0.014*ecP_sig*ecPhi_sig;
       V(3*i+3,3*i+2) = 0.0076*ecTh_sig*ecPhi_sig;
       V(3*i+3,3*i+3) = pow(ecPhi_sig,2);   
     }

     if(__multi==false &&  __p_names[i]!="gamma" && __p_names[i]!="n" ) {                                     
       if(__p4[i].M() < .7 || __p_names[i]=="pi+" || __p_names[i]=="pi-") part_index = 1; // use pion parameters
       else part_index = 0; // use proton parameters
       
       sector = GetSectorFromP4(__p4[i]);
       bin = GetThetaBin(theta);
       
       if(part_index  == 0 && bin > 10) bin = 10;
       
       // get resolution/eloss parameters
       p_res_scale = pPars[part_index][sector-1][bin][0];
       sigma_eloss = pPars[part_index][sector-1][bin][1];
       if(__isMC) p_res_scale *= 1.4;
       
       if(part_index == 0 && p_mag > 2.) p_res_scale *= 1.25;
       if(part_index == 0){
	 if(p_mag > 0.4) sigma_eloss *= 0.8;
	 else if(p_mag < 0.3) sigma_eloss *= 1.25;
       }
       p_res_scale*= 1.4*p_scale;
       // scale resolution errors
       V(3*i+1,3*i+1) *= pow(p_res_scale,2);

       // add in eloss errors
       if(beta < .765)
	 V(3*i+1,3*i+1) += pow(sigma_eloss*gamma/beta,2)*(1.-beta*beta/2.);
       else
	 V(3*i+1,3*i+1) += pow(sigma_eloss/.765,2)
	   *(1.-.765*.765/2.)/(1.-.765*.765);
       
       p_offset = pPars[part_index][sector-1][bin][2];
       V(3*i+1,3*i+1) += p_offset*p_offset + 2*p_offset*sqrt(V(3*i+1,3*i+1));
       
       // scale angular resolution errors
       lam_sig_ms = lamPars[part_index][sector-1][bin];
       phi_sig_ms = phiPars[part_index][sector-1][bin];
       
       V(3*i+2,3*i+2) *= pow(res_scale,2);
       V(3*i+3,3*i+3) *= pow(res_scale,2);
       
       // add in multiple scattering errors 
       V(3*i+2,3*i+2) += pow(lam_sig_ms/(p_mag*beta),2);
       V(3*i+3,3*i+3) += pow(phi_sig_ms/(p_mag*beta),2);
       
       // scale off diagnol elements by resolution scale factors
       V(3*i+1,3*i+2) *= res_scale*p_res_scale; 
       V(3*i+2,3*i+1) = V(3*i+1,3*i+2);
       V(3*i+1,3*i+3) *= res_scale*p_res_scale; 
       V(3*i+3,3*i+1) = V(3*i+1,3*i+3);
       V(3*i+2,3*i+3) *= res_scale*res_scale;
       V(3*i+3,3*i+2) = V(3*i+2,3*i+3);
     }
     
     if(__multi==true &&  __p_names[i]!="gamma" &&  __p_names[i]!="n" ){
       Float_t res_scaleLam,res_scalePhi;	 
       /*
	 These will probably change again as we learn more 
	 about how different particle respond
       */
       p_res_scale = 2.2*r_scale;
       res_scaleLam = res_scale*2.2*r_scale;	 
       res_scalePhi = res_scale*2.2*r_scale;       
       /*
	 if(__p_names[i]=="p"){
	 p_res_scale = 1.95;
	 res_scaleLam = res_scale*2.0;	 
	 res_scalePhi = res_scale*3.9;
	 }
	 else if(__p_names[i]=="pi+" ||"pi-" ||"pip"||"pim"){	 
	 p_res_scale = 1.95;
	 res_scaleLam = res_scale*2.0;	 
	 res_scalePhi = res_scale*3.9;	 
	 } 
       */       
       if(__isMC) p_res_scale *= 1.4;
       
       
       // scale angular resolution and energy loss using the lynch/Dhal formalism
       sthetsq = (__p4[i].Px()*__p4[i].Px() + __p4[i].Py()*__p4[i].Py())/(p_mag*p_mag);
       
       MultiScat(__p4[i],__vert[i],&ms,&el,cell);
             
       V(3*i+1,3*i+1) *= pow(p_res_scale,2);

       // add in eloss errors
       V(3*i+1,3*i+1) += el*0.3;
       //std::cout<<res_scalePhi<<std::endl;
       // scale res terms
       V(3*i+2,3*i+2) *= pow(res_scaleLam,2);
       V(3*i+3,3*i+3) *= pow(res_scalePhi,2);
       
       // angular multiple scattering
       // add in multiple scattering errors 
       V(3*i+2,3*i+2) +=ms*0.33;
       V(3*i+3,3*i+3) +=0.0033*ms/sthetsq;
       
       // scale off diagnol elements by resolution scale factors
       V(3*i+1,3*i+2) *= res_scaleLam*p_res_scale; 
       V(3*i+2,3*i+1) = V(3*i+1,3*i+2);
       V(3*i+1,3*i+3) *= res_scalePhi*p_res_scale; 
       V(3*i+3,3*i+1) = V(3*i+1,3*i+3);
       V(3*i+2,3*i+3) *= res_scalePhi*res_scaleLam;
       V(3*i+3,3*i+2) = V(3*i+2,3*i+3);		
     }
     
   }
   
   return V;
}
//_____________________________________________________________________________

/// Reads in the CLAS resolution parameters
bool ReadInResParams(double __pPars[2][6][15][3],double __lamPars[2][6][15],
		     double __phiPars[2][6][15],String &__xid){
  
  std::ifstream *inFile = 0;
  String fileName_base = "./parms/";
  String fileName;
  String p_names[2];
  p_names[0] = "p";
  p_names[1] = "pi";
  int num_rows_read,bin,sector;
  double par[3];
  
  for(int part = 0; part < 2; part++){
    // open |p| pars files
    if(__xid == "g11a") fileName = fileName_base + "g11parm_mom." + p_names[part];
    else if(__xid == "g10a") fileName = fileName_base + "g10parm_mom." + p_names[part];
    else if(__xid == "g12") fileName = fileName_base + "g12parm_mom." + p_names[part];
    else if(__xid == "g13a")fileName = fileName_base + "g13parm_mom." + p_names[part];
    inFile = new std::ifstream(fileName.c_str());		
    
    if(!(inFile->is_open())){
      // file didn't open
      std::cout << "Error! <ReadInResParams> File read error: " << fileName
		<< " Unable to open file. Make sure the correct parameters are available!!" << std::endl;
      return false;
    }
    
    num_rows_read = 0;
    while(*inFile >> sector){
      num_rows_read++;
      *inFile >> bin >> par[0] >> par[1] >> par[2];
      for(int i = 0; i < 3; i++) __pPars[part][sector-1][bin][i] = par[i];
    }
    
    // check correct number of rows were read in
    if(!CheckRowsRead(fileName,num_rows_read)) return false;
    delete inFile; inFile = 0;
    
    // read in lambda tracking angle pars
    if(__xid == "g11a") fileName = fileName_base + "g11parm_lam." + p_names[part];
    else if(__xid == "g10a") fileName = fileName_base + "g10parm_lam." + p_names[part];
    else if(__xid == "g12") fileName = fileName_base + "g12parm_lam." + p_names[part];
    else if(__xid == "g13a") fileName = fileName_base + "g13parm_lam." + p_names[part];
    inFile = new std::ifstream(fileName.c_str());
    
    if(!(inFile->is_open())){
      // file didn't open
      std::cout << "Error! <ReadInResParams> File read error: " << fileName
		<< " Unable to open file." << std::endl;
      return false;
    }
    
    num_rows_read = 0;
    while(*inFile >> sector){
      num_rows_read++;
      *inFile >> bin >> par[0];
      __lamPars[part][sector-1][bin] = par[0];
    }
    
    // check correct number of rows were read in
    if(!CheckRowsRead(fileName,num_rows_read)) return false;
    delete inFile; inFile = 0;
    
    // read in phi tracking angle pars
    
    if(__xid == "g11a") fileName = fileName_base + "g11parm_phi." + p_names[part];
    else if(__xid == "g10a") fileName = fileName_base + "g10parm_phi." + p_names[part];
    else if(__xid == "g12") fileName = fileName_base + "g12parm_phi." + p_names[part];
    else if(__xid == "g13a") fileName = fileName_base + "g13parm_phi." + p_names[part];
    inFile = new std::ifstream(fileName.c_str());     
    
    if(!(inFile->is_open())){ // file didn't open
      std::cout << "Error! <ReadInResParams> File read error: " << fileName
		<< " Unable to open file." << std::endl;
      return false;
    }
    
    num_rows_read = 0;
    while(*inFile >> sector){
      num_rows_read++;
      *inFile >> bin >> par[0];
      __phiPars[part][sector-1][bin] = par[0];
    }
    
    // check correct number of rows were read in
    if(!CheckRowsRead(fileName,num_rows_read)) return false;
    delete inFile; inFile = 0;
  }
  
  return true; // if we get here, everything should be ok
}
