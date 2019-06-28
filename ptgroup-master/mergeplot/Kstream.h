#ifndef _Kstream_H
#define _Kstream_H


#include <vector>
#include <string>
#include <iostream>
#include "TMatrixD.h"
#include "TMath.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TString.h"
#include "TPRegexp.h"
#include "KinLine.h"


typedef std::string String;

class Kstream {
  
private:
  // Data Members (private):
  
    /* names for particle id in kinematic fit  added for neutral test*/
  std::vector<String> _p_names;  ///<Particle names
  

  /* kinematic fitting statistical quantities */
  std::vector<double> _pulls; ///< Pull quantities of last fit
  double _chi2; ///< \f$ \chi^2 \f$ of last fit
  int _ndf; ///< Number of degrees-of-freedom of last fit

  /* kinematic quantities */
  double _ephot_in; /// < Photon energy (in)
  double _ephot_out; /// < Photon energy (out)
  std::vector<TLorentzVector> _p4in; ///< Particle 4-momenta (in)
  std::vector<TLorentzVector> _p4out; ///< Particle 4-momenta (out)
  double _targetMass; ///< Target mass

  /* Variance matrix info */
  TMatrixD _V; ///< Variance matrix
  double _sigma_missing[3]; ///< Fit errors on missing quantities

  /* missing particle info */
  double _missingMass; ///< Mass of the missing particle
  bool _misswitch; ///< Is there a missing particle?
  bool _missingNeutral;  ///< Is the missing particle a neutral?  ADDed for neutral

  /* extra mass constraint info */
  bool _extraC; ///< Is there an extra mass constraint?
  double _invariantMass; ///< Invariant mass used in extra mass constraint
  std::vector<bool> _extraC_meas; ///< Which measured particles in constraint?
  bool _extraC_miss; ///< Is missing particle used in extra mass constraint?

  // Functions (private):

  void _MainFitter();

  void _ResetForNewFit();

  void _Copy(const Kstream &__kfit);

  void _ZeroOutFit();

  void _SetMissingParticleErrors(const TMatrixD &__missingV,
				 const TVectorD &__x);


public:
  // Create/Copy/Destroy:
  Kstream();

  Kstream(const Kstream &__kfit){ 
    /// Copy Constructor
    this->_Copy(__kfit);
  }

  virtual ~Kstream(){ 
    /// Destructor
    _pulls.clear();
    _p4in.clear();
    _p4out.clear();
    _extraC_meas.clear();
  }

  Kstream& operator=(const Kstream &__kfit){
    /// Assignment operator
    this->_Copy(__kfit);
    return *this;
  }

  // Setters:
  
  //added for neutrals  
    
  inline void StringNames(const std::vector<String> &__p_names) {
  _p_names = __p_names;
  }  

  inline void SetVMatrix(const TMatrixD &__VMat){ 
    /// Set the Variance matrix.
    _V.ResizeTo(__VMat);
    _V = __VMat;
  }

  inline void SetP4(const std::vector<TLorentzVector> &__p4){
    /// Set the input 4-momenta.
    _p4in = __p4;
  }
  
  inline void SetPhotonEnergy(double __erg){ 
    /// Set the input tagged photon energy.
    _ephot_in = __erg;
  }

  inline void SetTargetMass(double __mass){
    /// Set the target mass (the ctor sets this to be \f$ m_p \f$)
    _targetMass = __mass;
  }

  inline void FitInput(double __ephot,const std::vector<TLorentzVector> &__p4,
		       const TMatrixD &__VMat,double __targMass = -1.){
    if(__targMass > 0.) _targetMass = __targMass;
    _ephot_in = __ephot;
    _p4in = __p4;
    _V.ResizeTo(__VMat);
    _V = __VMat;
  }

  inline double Chi2() const {
    /// Return \f$ \chi^2 \f$ of the last fit.
    return _chi2;
  }

  inline int Ndf() const {
    /// Return the number of degrees of freedom of the last fit
    return _ndf;
  }

  inline double GetPull(int __n) const {
    /// Returns the @a n pull quantity
    return _pulls[__n];
  }

  inline const TLorentzVector& FitP4(int __n) const {
    return _p4out[__n];
  }

  inline double FitPhotonEnergy() const {
    return _ephot_out;
  }

  inline double GetMissingError(int __n) const {
    return _sigma_missing[__n];
  }

  inline const TMatrixD& GetVMat() const {
    return _V;
  }

  // Functions:

  double Prob() const {
    return TMath::Prob(_chi2,_ndf);
  }
  static double NameToMass(const String &__name);
  double Fit(const String &__miss,
	     const std::vector<bool> &__constrain_meas = std::vector<bool>(),
	     bool __constrain_miss = false,double __invariantMass = -1.);

  double Fit(double __missingMass = -1.,
	     const std::vector<bool> &__constrain_meas = std::vector<bool>(),
	     bool __constrain_miss = false,double __invariantMass = -1.);

};
//_____________________________________________________________________________

#endif /* _Kstream_H */


