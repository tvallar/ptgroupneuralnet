      SUBROUTINE STCOUNTER(icell, vertex, cdir, dist, xpoint)
c
c_begin_doc
c  RCS ID string
c  $Id: stcounter.F,v 1.8 2006/01/10 22:40:26 hovanes Exp $
c
c  Documentation for subroutine STCOUNTER
c
c  Purpose: Calculate track length within the start counter scintillator
c  --------
c
c  Input Parameters:  (Name - Type - Meaning)
c  ----------------
c  vertex(3) (real) vertex coordinates
c  cdir(3)   (real) track direction cosines  
c
c  Output Parameters:  (Name - Type - Meaning)
c  -----------------
c  dist (real) - track length within the scintillator
c  xpoint(3) - intercept of a track with start counter
c
c  Other routines:
c  ---------------
c  IntesectionPlan 
c
c  Notes:
c  ------
c
c  Author:   Eugene Pasyuk      Created:  Tue Apr  6 16:38:45 EDT 1999
c  -------
c
c  Major revisions:
c  ----------------
c     
c
c_end_doc
c
      IMPLICIT NONE
      SAVE 
c
c_begin_var
c  input/output variables:
c  -----------------------
      INTEGER icell
      REAL vertex(3)
      REAL cdir(3)
      REAL dist
      REAL xpoint(3)
c
c  Local pre-defined variables:
c  ---------------------------

      REAL d_leg                !distance from beam axis to the 
                                !face of scintillator
      REAL thickness            !scintillator thickness
      REAL half_length          !half length of the leg
      REAL pi

      DATA d_leg/10.2/
      DATA thickness/0.32/      
c changed thickness from 0.3 to 0.32 per Gordon's estimate of wrapping maerial
      DATA half_length/20.0/
      DATA pi/3.14159265359/
c  RCS information: 
      CHARACTER*132  CFILE, CREVIS, CSTATE, CDATE, CAUTHO, CRCSID
      PARAMETER (CFILE = '$RCSfile: stcounter.F,v $')
      PARAMETER (CREVIS = '$Revision: 1.8 $')
      PARAMETER (CSTATE = '$State: Exp $')
      PARAMETER (CDATE = '$Date: 2006/01/10 22:40:26 $')
      PARAMETER (CAUTHO = '$Author: hovanes $')
      DATA CRCSID/   
     1'$Id: stcounter.F,v 1.8 2006/01/10 22:40:26 hovanes Exp $'   
     2/   
c  Module information:
      CHARACTER*(*)  CRNAME, CRAUTH
      CHARACTER*100  CRMESS
      PARAMETER (CRNAME='STCOUNTER')
      PARAMETER (CRAUTH='Eugene Pasyuk')
c
c  Local User defined variables:
c  -----------------------------
      INTEGER NWRITE
      integer i
      integer ierr
      REAL alpha
      REAL w
      REAL phi 
      REAL point(3)
      REAL np(3)
      REAL start(3)
      REAL theta
      REAL dir(3)
      REAL aupoint(3)
      REAL norme
      REAL csinus
      REAL sind, cosd, angle
      DATA NWRITE/0/
      
      sind(angle) = sin( angle*pi/180. )
      cosd(angle) = cos( angle*pi/180. )

#ifdef Linux
c for Linux these functions are not implemeted in standard library

      sind(angle) = sin( angle*pi/180. )
      cosd(angle) = cos( angle*pi/180. )

#endif

#include "eloss_geom.PAR"


c_end_var
c
c  executable code for routine STCOUNTER:
c----6----------------------------------------------------------------72
c
      IF (NWRITE .LT. 1) THEN
        NWRITE = NWRITE + 1
         write(CRMESS,*) 
     + 'STCOUNTER: ST offset is: ', st_offset
c        CRMESS='This is a STCOUNTER routine, this message written once'
c        CALL RECMES(CRNAME,'I',CRMESS)
      ENDIF

c
c-------------Intersection avec le Start Counter
c

      if( icell .GE. 7 ) then                              ! g11 and above
        d_leg = 10.315                                     ! g6->/10.2/
        half_length=26.15                                  ! g6->/20.0/
        thickness = 0.215                  ! plastic scintillator (rho = 1.032 g/cm^2)
     &            + 0.027*2*1.020/1.032    ! photographic tape (rho = 1.02 g/cm^2)
     &            + 0.010  *1.150/1.032    ! tedlar (rho = 1.15 g/cm^2)
     &            + 0.0013 *1.300/1.032    ! VM2000 (rho = 1.30 g/cm^2)
     &            + 0.975  *0.031/1.032    ! outer foam shell (rho = 30.9 mg/cm^2)
     &            + 0.571  *0.110/1.032    ! support foam shell (rho = 110. mg/cm^2)
      endif

      alpha = atan2(cdir(2),cdir(1))
      alpha = 3.*alpha/pi
      w = nint(alpha)
      phi = w*60.

      np(1) = cosd(phi)
      np(2) = sind(phi)
      np(3) = 0.

      do i = 1,3
         point(i) = vertex(i) + 500.*cdir(i)
         start(i) = d_leg*np(i)
      enddo

      call IntersectionPlan(vertex, point, start, np, aupoint, ierr)
 
      norme = 0.
      do i = 1,3
         xpoint(i) = aupoint(i)
         norme = norme+(aupoint(i)-vertex(i))*(aupoint(i)-vertex(i))
      enddo
      norme = sqrt(norme)
 
      theta = 90.
      
      if ( aupoint(3) .gt. (half_length + st_offset) ) then !leg or nose?
         theta = 60.
         if( icell .GE. 7 ) theta = 45.
      endif

      dir(1) = sind(theta)*cosd(phi)
      dir(2) = sind(theta)*sind(phi)
      dir(3) = cosd(theta)
     
      csinus = 0.
      do i = 1,3
         csinus = csinus+(aupoint(i)-vertex(i))*dir(i)
      enddo

      csinus = csinus/norme
      dist = thickness/csinus

      if ( aupoint(3) .gt. (half_length + st_offset) ) then
c     -- nose --
         call IntersectionPlan(vertex, point, start, dir, aupoint, ierr)
         do i = 1,3
            xpoint(i) = aupoint(i)
         enddo
      endif

      RETURN
      END
c
c------------------------------------------------------------------------------




