#include "Kstream.h"
#include "TVectorD.h"
#include "TArrayD.h"
#include "TMath.h"
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


Kstream::Kstream(){
  _targetMass = 0.938272;
  _ndf = 0;
  _chi2 = 200.;
  _ephot_in = 0.;
  _ephot_out = 0.;
  _missingMass = -1.;
  _misswitch = false;
  _missingNeutral = false; 
  _extraC = false;
  _invariantMass = -1.;
  _extraC_miss = false;
}

void Kstream::_Copy(const Kstream &__stream){
  _pulls = __stream._pulls;
  _chi2 = __stream._chi2;
  _ndf = __stream._ndf;
  _ephot_in = __stream._ephot_in;
  _ephot_out = __stream._ephot_out;
  _p4in = __stream._p4in;
  _p4out = __stream._p4out;
  _targetMass = __stream._targetMass;
  _V.ResizeTo(__stream._V);
  _V = __stream._V;
  _missingMass = __stream._missingMass;
  _misswitch = __stream._misswitch;
  _extraC = __stream._extraC;
  _invariantMass = __stream._invariantMass;
  _extraC_meas = __stream._extraC_meas;
  _extraC_miss = __stream._extraC_miss;
  _p_names = __stream._p_names; 	
  for(int i = 0; i < 3; i++) _sigma_missing[i] = __stream._sigma_missing[i];
}

void Kstream::_MainFitter(){
  const int numParts = (int)_p4in.size();
  const int dim = 3*numParts + 1; // number of measured quantites
  if(_V.GetNrows() != dim || _V.GetNcols() != dim){
    std::cout << "Error! <Kstream::_MainFitter> Variance matrix size is NOT "
	      << "correct for current number of particles. For " << numParts
	      << " particles, the Variance matrix should be " << dim << " x "
	      << dim << ", but the matrix passed in is " << _V.GetNrows()
	      << " x " << _V.GetNcols() << std::endl;
    abort();
  }

  int numConstraints = 4;
  if(_extraC) numConstraints = 5;

  _ndf = 4;
  if(_misswitch) _ndf -= 3;
  if(_extraC) _ndf += 1;

  int i,j;
  
  double theta[numParts]; //for neutrals
    
  double alpha[numParts],mass[numParts],lambda[numParts],phi[numParts];
  double p[numParts],erg[numParts],px[numParts],py[numParts],pz[numParts];
  double e_miss=0.,e_inv=0.,px_inv=0.,py_inv=0.,pz_inv=0.;
  
  TVector3 p3_miss;    
  TVectorD initialy(dim),y(dim);  
  TVectorD xi(3),x(3),delx(3);  
  TMatrixD Fmatrix(numConstraints,3),FmatrixT(3,numConstraints);
  TMatrixD Gmatrix(numConstraints,dim),GmatrixT(dim,numConstraints);
  TVectorD ConMatrix(numConstraints);  
  TVectorD eps(dim);  
  TMatrixD epsmatrix(1,dim), epsmatrixT(dim,1);  
  TVectorD delta(dim);  
  TVectorD xsi(3);   
  TMatrixD gGmatrix(numConstraints,numConstraints);
  TMatrixD V_fit(dim,dim);
  TMatrixD V_improved(dim,dim);
  
  initialy(0) = _ephot_in;
  for(i = 0; i < numParts; i++){
    if(_p_names[i] == "n" or _p_names[i] == "gamma") {
	initialy(3*i+1) = _p4in[i].P();
	initialy(3*i+2) = _p4in[i].Theta();
	initialy(3*i+3) = _p4in[i].Phi();	
	if(_p_names[i] == "gamma") mass[i] = 0.0000000;
	}    
   else {
	initialy(3*i+1) = _p4in[i].P(); 
	initialy(3*i+2) = LambdaTrack(_p4in[i]); 
	initialy(3*i+3) = PhiTrack(_p4in[i]); 
	alpha[i] = AlphaTrack(_p4in[i]); 
	mass[i] = _p4in[i].M(); 
	}
  }
  y = initialy; 
  if(_misswitch){
    xi(0) = 0.; // px
    xi(1) = 0.; // py
    xi(2) = _ephot_in; // pz
    for(i = 0; i < numParts; i++){
      xi(0) -= _p4in[i].Px();
      xi(1) -= _p4in[i].Py();
      xi(2) -= _p4in[i].Pz();
    }
  }

  x = xi;  

  for(int iter = 0; iter < 10; iter++){
    Fmatrix.Zero();
    FmatrixT.Zero();
    Gmatrix.Zero();
    GmatrixT.Zero();
    ConMatrix.Zero();
    delta.Zero();
    xsi.Zero();
  
    for(i = 0; i < numParts; i++){
	if(_p_names[i] == "n" or _p_names[i] == "gamma") {
	p[i] = y(3*i+1);
	theta[i] = y(3*i+2);
	phi[i] = y(3*i+3);
	erg[i] = sqrt(p[i]*p[i] + mass[i]*mass[i]);
	px[i] = p[i]*sin(theta[i])*cos(phi[i]);
	py[i] = p[i]*sin(theta[i])*sin(phi[i]);
	pz[i] = p[i]*cos(theta[i]);
 	} 
	else {   
        lambda[i] = y(3*i+2);
        phi[i] = y(3*i+3);
        p[i] = y(3*i+1);
        erg[i] = sqrt(p[i]*p[i] + mass[i]*mass[i]);
        px[i] = p[i]*(cos(lambda[i])*sin(phi[i])*cos(alpha[i]) 
		    - sin(lambda[i])*sin(alpha[i]));
        py[i] = p[i]*(cos(lambda[i])*sin(phi[i])*sin(alpha[i]) 
		    + sin(lambda[i])*cos(alpha[i]));
        pz[i] = p[i]*cos(lambda[i])*cos(phi[i]);
           }
	}
    if(_misswitch){
      p3_miss.SetXYZ(x(0),x(1),x(2)); // p3 missing
      e_miss = sqrt(p3_miss.Mag2() + pow(_missingMass,2)); // energy missing
    }
    ConMatrix(0) = y(0) + _targetMass;
    ConMatrix(3) = -y(0);
    for(i = 0; i < numParts; i++){
      ConMatrix(0) -= erg[i];
      ConMatrix(1) += px[i];
      ConMatrix(2) += py[i];
      ConMatrix(3) += pz[i];
    }
    if(_misswitch){
      ConMatrix(0) -= e_miss;
      ConMatrix(1) += p3_miss.X();
      ConMatrix(2) += p3_miss.Y();
      ConMatrix(3) += p3_miss.Z();
    }
    
    if(_extraC){
      e_inv = 0.;
      px_inv = 0.;
      py_inv = 0.;
      pz_inv = 0.;
      for(i = 0; i < numParts; i++){
	if(_extraC_meas[i]){
	  e_inv += erg[i];
	  px_inv += px[i];
	  py_inv += py[i];
	  pz_inv += pz[i];
	}
      }
      if(_extraC_miss){
	e_inv += e_miss;
	px_inv += p3_miss.X();
	py_inv += p3_miss.Y();
	pz_inv += p3_miss.Z();	
      }
   
      ConMatrix(4) = e_inv*e_inv - px_inv*px_inv - py_inv*py_inv - pz_inv*pz_inv - 				     pow(_invariantMass,2);  
    }
    if(_misswitch){
      for(i = 0; i < 3; i++){
	Fmatrix(0,i) = -x(i)/e_miss; 
	Fmatrix(i+1,i) = 1.; 
      }
      if(_extraC && _extraC_miss){
	Fmatrix(4,0) = 2.*(e_inv*(x(0)/e_miss) - px_inv);
	Fmatrix(4,1) = 2.*(e_inv*(x(1)/e_miss) - py_inv);
	Fmatrix(4,2) = 2.*(e_inv*(x(2)/e_miss) - pz_inv);
      }
    }
    FmatrixT.Transpose(Fmatrix);
    Gmatrix(0,0) = 1.;
    Gmatrix(3,0) = -1;    
    for(i = 0; i < numParts; i++){
    
    if(_p_names[i] == "n" or _p_names[i] == "gamma") {
	Gmatrix(0,3*i+1) = -p[i]/erg[i];
	Gmatrix(1,3*i+1) = sin(theta[i])*cos(phi[i]);
	Gmatrix(2,3*i+1) = sin(theta[i])*sin(phi[i]);
	Gmatrix(3,3*i+1) = cos(theta[i]);
	Gmatrix(0,3*i+2) = 0.;
	Gmatrix(1,3*i+2) = p[i]*cos(theta[i])*cos(phi[i]);
	Gmatrix(2,3*i+2) = p[i]*cos(theta[i])*sin(phi[i]);
	Gmatrix(3,3*i+2) = -p[i]*sin(theta[i]);
	Gmatrix(0,3*i+3) = 0.;
	Gmatrix(1,3*i+3) = -p[i]*sin(theta[i])*sin(phi[i]);
	Gmatrix(2,3*i+3) = p[i]*sin(theta[i])*cos(phi[i]);
	Gmatrix(3,3*i+3) = 0.;
	}    
   else {    
       Gmatrix(0,3*i+1) = -p[i]/erg[i];
       Gmatrix(1,3*i+1) = cos(lambda[i])*sin(phi[i])*cos(alpha[i]) 
	- sin(lambda[i])*sin(alpha[i]);
       Gmatrix(2,3*i+1) = cos(lambda[i])*sin(phi[i])*sin(alpha[i]) 
	+ sin(lambda[i])*cos(alpha[i]);
       Gmatrix(3,3*i+1) = cos(lambda[i])*cos(phi[i]);
       Gmatrix(0,3*i+2) = 0.;
       Gmatrix(1,3*i+2) = p[i]*(-sin(lambda[i])*sin(phi[i])*cos(alpha[i])
			 - cos(lambda[i])*sin(alpha[i]));
       Gmatrix(2,3*i+2) = p[i]*(-sin(lambda[i])*sin(phi[i])*sin(alpha[i])
			 + cos(lambda[i])*cos(alpha[i]));
       Gmatrix(3,3*i+2) = p[i]*(-sin(lambda[i])*cos(phi[i]));
       Gmatrix(0,3*i+3) = 0.;
       Gmatrix(1,3*i+3) = p[i]*cos(lambda[i])*cos(phi[i])*cos(alpha[i]);
       Gmatrix(2,3*i+3) = p[i]*cos(lambda[i])*cos(phi[i])*sin(alpha[i]);
       Gmatrix(3,3*i+3) = p[i]*cos(lambda[i])*(-sin(phi[i]));
	}
      if(_extraC_meas[i]){
	for(j = 1; j <= 3; j++){
	  int ind = 3*i + j;
	  Gmatrix(4,ind) = 2.*(-e_inv*Gmatrix(0,ind) - px_inv*Gmatrix(1,ind) - py_inv*Gmatrix(2,ind) 
			 - pz_inv*Gmatrix(3,ind));
	}
      }	      
    }
    GmatrixT.Transpose(Gmatrix);
    gGmatrix = Gmatrix * _V * GmatrixT;  
    gGmatrix.Invert();

    if(_misswitch){
      xsi = ((FmatrixT * gGmatrix * Fmatrix).Invert())*(FmatrixT * gGmatrix * ConMatrix); 
      x -= xsi;
      delta = _V * GmatrixT * gGmatrix * (ConMatrix - Fmatrix*xsi);
    }
    else delta = _V * GmatrixT * gGmatrix * ConMatrix;
    y -= delta;
    eps = initialy - y;
    TMatrixDColumn(epsmatrixT,0) = eps;
    epsmatrix.Transpose(epsmatrixT);
    _chi2 = (epsmatrix * GmatrixT * gGmatrix * Gmatrix * epsmatrixT)(0,0); 
  
  } 

    eps = initialy - y;
    TMatrixDColumn(epsmatrixT,0) = eps;
  if(_misswitch){
    V_fit = _V - _V*GmatrixT*gGmatrix*Gmatrix*_V 
      + _V*GmatrixT*gGmatrix*Fmatrix*((FmatrixT*gGmatrix*Fmatrix).Invert())*FmatrixT*gGmatrix*Gmatrix*_V;
  }
  else 
    V_fit = _V - _V*GmatrixT*gGmatrix*Gmatrix*_V;
  _pulls.resize(dim);
  for(i = 0; i < dim; i++)
    _pulls[i] = -eps(i)/sqrt(_V(i,i) - V_fit(i,i));
    _chi2 = (epsmatrix * GmatrixT * gGmatrix * Gmatrix * epsmatrixT)(0,0); 
   _p4out.resize(numParts);
  for(i = 0; i < numParts; i++) _p4out[i].SetPxPyPzE(px[i],py[i],pz[i],erg[i]);
  _ephot_out = y(0);

  TLorentzVector MP4 =_p4in[1] - _p4out[1] - _p4out[2];
  
  if(_misswitch)
    this->_SetMissingParticleErrors((FmatrixT * gGmatrix * Fmatrix).Invert(),x); 
}

void Kstream::_ResetForNewFit(){
  int size = (int)_p4in.size();
  _p4out.resize(size);
  _extraC_meas.resize(size);
  _pulls.resize(3*size + 1);
  
  _misswitch = false;
  _extraC_miss = false;
  for(int i = 0; i < size; i++) _extraC_meas[i] = false;
  for(int i = 0; i < 3; i++) _sigma_missing[i] = 0.;
}

double Kstream::Fit(double __missingMass,
		   const std::vector<bool> &__constrain_meas,
		   bool __constrain_miss,double __invariantMass){

  this->_ResetForNewFit();

  // check for missing particle
  if(__missingMass >= 0.){
    // there is a missing particle and this is its mass
    _missingMass = __missingMass;
    _misswitch = true;
  }
   
  // check for extra mass constraint
  if(__invariantMass > 0.){
    _invariantMass = __invariantMass;
    _extraC = true;
    _extraC_miss = __constrain_miss;
    if(__constrain_meas.size() == _extraC_meas.size())
      _extraC_meas = __constrain_meas;
    else{
      std::cout << "Error! <Kstream::Fit> Boolean vector (argument 2) passed to"
		<< " this function has incorrect size. The fit will NOT be "
		<< "run." << std::endl;
      this->_ZeroOutFit();
      return 0.;
    }
  }
 
  // run the fit
  this->_MainFitter();

  // return the confidence level
  return this->Prob();
}

double Kstream::Fit(const String &__miss,
		   const std::vector<bool> &__constrain_meas,
		   bool __constrain_miss,double __invariantMass){ 
  double mass = -1.;
  if(__miss != String()){
    // a missing particle has been specified
    mass = this->NameToMass(__miss);
  }
    
  this->Fit(mass,__constrain_meas,__constrain_miss,__invariantMass);
  return this->Prob();
}

double Kstream::NameToMass(const String &__name) {
  if(__name == String()) return -1.;

  String name = __name;
  // make all letters lower case
  transform(name.begin(),name.end(),name.begin(),(int(*)(int)) tolower);
  // remove any white space, -'s or _'s
  for(String::size_type pos = 0; pos < name.size(); pos++){
    if(name[pos] == ' ' || name[pos] == '-' || name[pos] == '_'){
      name.erase(pos,1);
      pos--;
    }
  }
  if(name == "gamma") return 0.00000000000000000; 
  if(name == "pi" || name == "pi+" || name == "pi-") return 0.13957;
  if(name == "pi0") return 0.13498;
  if(name == "k" || name == "k+" || name == "k-") return 0.493677;
  if(name == "k0" || name == "kshort" || name == "klong") return 0.497672;
  if(name == "eta") return 0.54730;
  if(name == "rho" || name == "rho+" || name == "rho-" || name == "rho0")
    return 0.7693;
  if(name == "omega") return 0.78257;
  if(name == "eta'" || name == "etaprime") return 0.95778;
  if(name == "phi") return 1.019417;
  if(name == "p" || name == "proton" || name == "antiproton" || name == "pbar"
     || name == "p-" || name == "p+") return 0.93827;
  if(name == "neutron" || name == "n") return 0.93957;
  std::cout << "Warning! <Kstream::NameToMass> Unknown particle name: " 
	    << __name << ". Mass will be returned as 0." << std::endl;
  return 0.;
}

void Kstream::_ZeroOutFit(){
  for(int i = 0; i < (int)_pulls.size(); i++) _pulls[i] = 0.;
  _chi2 = 0.;
  _ndf = 0;
  _ephot_out = 0.;
  for(int i = 0; i < (int)_p4out.size(); i++) _p4out[i].SetXYZT(0.,0.,0.,0.);  
}

void Kstream::_SetMissingParticleErrors(const TMatrixD &__missingV,
				       const TVectorD &__x){

  double p = sqrt(__x(0)*__x(0) + __x(1)*__x(1) + __x(2)*__x(2));
  TLorentzVector p4(__x(0),__x(1),__x(2),sqrt(p*p + pow(_missingMass,2)));

  // kinematic quantities in tracking coordinates
  double phi = PhiTrack(p4);
  double lam = LambdaTrack(p4);
  double alpha = AlphaTrack(p4);

  // missing particle errors in lab coordinates
  double sigma_px_2 = __missingV(0,0);
  double sigma_py_2 = __missingV(1,1);
  double sigma_pz_2 = __missingV(2,2);
  
  // jacobian elements we need
  double dp_dpx = __x(0)/p;
  double dp_dpy = __x(1)/p;
  double dp_dpz = __x(2)/p;

  double dlam_dpx = (-sin(alpha) - sin(lam)*dp_dpx)/(p*cos(lam));
  double dlam_dpy = (cos(alpha) - sin(lam)*dp_dpy)/(p*cos(lam));
  double dlam_dpz = -tan(lam)*dp_dpz/p;

  double dphi_dpx = (sin(lam)*p*cos(phi)*dlam_dpx - dp_dpx*cos(lam)*cos(phi))
    /(-sin(phi)*p*cos(lam));
  double dphi_dpy = (sin(lam)*p*cos(phi)*dlam_dpy - dp_dpy*cos(lam)*cos(phi))
    /(-sin(phi)*p*cos(lam));
  double dphi_dpz = (1 + sin(lam)*p*cos(phi)*dlam_dpz 
		     - dp_dpz*cos(lam)*cos(phi))/(-sin(phi)*p*cos(lam));

  // get error on p
  _sigma_missing[0] = sqrt(sigma_px_2*(dp_dpx*dp_dpx) 
			   + sigma_py_2*(dp_dpy*dp_dpy)
			   + sigma_pz_2*(dp_dpz*dp_dpz));

  // get error on lambda
  _sigma_missing[1] = sqrt(sigma_px_2*(dlam_dpx*dlam_dpx) 
			   + sigma_py_2*(dlam_dpy*dlam_dpy)
			   + sigma_pz_2*(dlam_dpz*dlam_dpz));

  // get error on phi
  _sigma_missing[2] = sqrt(sigma_px_2*(dphi_dpx*dphi_dpx) 
			   + sigma_py_2*(dphi_dpy*dphi_dpy)
			   + sigma_pz_2*(dphi_dpz*dphi_dpz));

}
