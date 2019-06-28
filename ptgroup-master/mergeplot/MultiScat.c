


extern"C" {
void ccylinder_(float *vert,float *cdir,float *dist);
void stcounter_(int *icell,float *vert,float *cdir,float *dist3,float *xpoint);
void targcell_(int *icell,float *vert,float *cdir,float *dist1,float *dist2);
}




void MultiScat(TLorentzVector P4in, TVector3 Vin, float *msq, float *usq,int icell){
  
  float cdir[3], vert[3],xpoint[3];
  float dist=0, dist1, dist2,dist3;
  float F,chi2c,chi2a,Omega,v,Z,A,rho,ms_lynch,beta=0;  
  float Amol,A_[3],Z_[3],percentage[3]; 
  float sig2p,pmag,mass, betasq,pmom,psq,X0,msq_old=0; 
  int i;
  float power = 0.6666666666667;
  vert[0] = Vin.X();
  vert[1] = Vin.Y();
  vert[2] = Vin.Z();
  cdir[0] = P4in.Px()/P4in.P();
  cdir[1] = P4in.Py()/P4in.P();
  cdir[2] = P4in.Py()/P4in.P();
  pmag = P4in.P();
  pmom = 1000.0*P4in.P();
  mass = 1000.0*P4in.M();
  *usq=0.;
  *msq=0.;
  psq = pmom*pmom;
  betasq = psq*psq/(psq + mass*mass);
 
  // Carbon beam pipe   
  ccylinder_(vert,cdir,&dist);
 
  X0 = 18.8;

  beta=pmom/sqrt(psq+mass*mass);

  F=0.99; // Fraction of the Moliere distribution to include for Multiple
	   // Scattering 

if (dist>0){
    // This is the Lynch/Dahl formalism for <theta_ms^2>.  
    //   See Leo, 2nd. Edition, p.47. 
    Z=6.;
    A=12.01;
    rho=1.7; // From loss.F comments 
    chi2c=0.157*Z*(Z+1.)/A*dist*rho/psq/beta/beta;
    chi2a=0.00002007*pow((float)Z,power)*(1.+3.34*(Z/137./beta)*(Z/137./beta))
      /psq;
    Omega=chi2c/chi2a;
    v=0.5*Omega/(1.-F);

    ms_lynch=2.*chi2c/(1.+F*F)*((1.+v)/v*log(1.+v)-1.);
     
    // Energy spread assuming thick absorber (Gaussian) limit.
    //  See Leo, 2nd. Edition, p. 49 //
    sig2p=0.1569*rho*Z/A*dist*(1.-beta*beta/2.)/(1.-beta*beta)/1000./1000.
      /pmag/pmag/pmag/pmag/beta/beta;
    
    msq_old = 184.96*dist/(betasq*X0);
    
    *usq=sig2p;    
    *msq=ms_lynch;
  }
  
 
  // Target cell 
  targcell_(&icell,vert,cdir,&dist1,&dist2);
  
  X0 = 865.0;  // Hydrogen
  
  if (dist1>0){
    A=1.01;
    Z=1.;
    rho=0.0708;
    chi2c=0.157*Z*(Z+1.)/A*dist1*rho/psq/beta/beta;
    chi2a=0.00002007*pow((float)Z,power)*(1.+3.34*(Z/137./beta)*(Z/137./beta))
      /psq;
    Omega=chi2c/chi2a;
    v=0.5*Omega/(1.-F);
    
    ms_lynch=2.*chi2c/(1.+F*F)*((1.+v)/v*log(1.+v)-1.);
    
    msq_old += 184.96*dist1/(betasq*X0);
    
    sig2p=0.1569*rho*Z/A*dist1*(1.-beta*beta/2.)/(1.-beta*beta)/1000./1000.
      /pmag/pmag/pmag/pmag/beta/beta;
   
    (*usq)+=sig2p; 
    (*msq)+=ms_lynch;
  }
  
  X0 = 28.7;
  
  // Mylar 
  if (dist2>0){                                              
    A_[0]=12.01;
    A_[1]=1.01;
    A_[2]=16.00;
    Z_[0]=6.;
    Z_[1]=1.;
    Z_[2]=8.;
    Z=A=0.;
    Amol=5.*A_[0]+4.*A_[1]+2.*A_[2];
    percentage[0]=5.*A_[0]/Amol;
    percentage[1]=4.*A_[1]/Amol;
    percentage[2]=2.*A_[2]/Amol;
    for (i=0;i<3;i++){
      Z+=percentage[i]*Z_[i];
      A+=percentage[i]*A_[i];
    }
    rho=1.39;
    chi2c=0.157*Z*(Z+1.)/A*dist2*rho/psq/beta/beta;
    chi2a=0.00002007*pow((float)Z,power)*(1.+3.34*(Z/137./beta)*(Z/137./beta))
      /psq;
    Omega=chi2c/chi2a;
    v=0.5*Omega/(1.-F);
    
    ms_lynch=2.*chi2c/(1.+F*F)*((1.+v)/v*log(1.+v)-1);    
    
    msq_old += 184.96*dist2/(betasq*X0); 
    
    sig2p=0.1569*rho*Z/A*dist2*(1.-beta*beta/2.)/(1.-beta*beta)/1000./1000.
      /pmag/pmag/pmag/pmag/beta/beta;
   
    (*usq)+=sig2p;
    
    (*msq)+=ms_lynch;
    
  }
  
   // Start Counter scintillator material 
  stcounter_(&icell,vert,cdir,&dist3,xpoint);

  X0 = 42.4;

  if (dist3<0.){
    // There's a flaw in the eloss code... 
  }
  if (dist3>0){
    A_[0]=12.01; 
    A_[1]=1.01;
    Z_[0]=6.;
    Z_[1]=1.;
    Z=A=0.;
    Amol=8.*A_[0]+8.*A_[1];  // Use polystyrene for scintillator 
    percentage[0]=8.*A_[0]/Amol;
    percentage[1]=8.*A_[1]/Amol;
    for (i=0;i<2;i++){
    Z+=percentage[i]*Z_[i];
    A+=percentage[i]*A_[i];
    }
    rho=1.032;
    chi2c=0.157*Z*(Z+1.)/A*dist3*rho/psq/beta/beta;
    chi2a=0.00002007*pow((float)Z,power)*(1.+3.34*(Z/137./beta)*(Z/137./beta))
      /psq;
    Omega=chi2c/chi2a;
    v=0.5*Omega/(1.-F);
    
    ms_lynch=2.*chi2c/(1.+F*F)*((1.+v)/v*log(1.+v)-1);
    
    msq_old += 184.96*dist3/(betasq*X0);
    
    sig2p=0.1569*rho*Z/A*dist3*(1.-beta*beta/2.)/(1.-beta*beta)/1000./1000.
      /pmag/pmag/pmag/pmag/beta/beta;
    
    (*usq)+=sig2p;
    
    (*msq)+=ms_lynch; 
      
  } 
    
}
