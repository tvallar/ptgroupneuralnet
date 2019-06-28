      SUBROUTINE CCYLINDER(  vertex, cdir, dist )
c
c_begin_doc
c  RCS ID string
c  $Id: ccylinder.F,v 1.7 2004/12/10 20:15:30 baltzell Exp $
c
c  Documentation for subroutine CCYLINDER
c
c  Purpose: Calcualte the ranage within carbon cylinder
c  --------
c
c  Input Parameters:  (Name - Type - Meaning)
c  ----------------
c  vertex(3) (real) vertex coordinates
c  cdir(3)   (real) track direction cosines  
c
c  Output Parameters:  (Name - Type - Meaning)
c  -----------------
c  dist (real) track length within the scintillator
c
c  Other routines:
c  ---------------
c  IntersectionCylindre
c
c  Notes:
c  ------
c
c  Author:   Eugene Pasyuk      Created:  Tue Apr  6 16:39:04 EDT 1999
c  -------
c
c  Major revisions:
c  ----------------
c  substituted IntersectionCylindre with InterCylinder 
c  (Dec. 6, 2002, Niccolai/Strauch)
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
      REAL vertex(3)
      REAL cdir(3)
      REAL dist

c  Local pre-defined variables:
c  ---------------------------
      REAL centre(3)            
      REAL rayon                !radius of the pipe
      REAL thickness            !thicknes of the pipe
      REAL extremtube           !half length
      REAL alpha                !cone angle
      REAL dir(3)               !direction of pipe axis
      REAL pi
      
c  RCS information: 
      CHARACTER*132  CFILE, CREVIS, CSTATE, CDATE, CAUTHO, CRCSID
      PARAMETER (CFILE = '$RCSfile: ccylinder.F,v $')
      PARAMETER (CREVIS = '$Revision: 1.7 $')
      PARAMETER (CSTATE = '$State: Exp $')
      PARAMETER (CDATE = '$Date: 2004/12/10 20:15:30 $')
      PARAMETER (CAUTHO = '$Author: baltzell $')
#include "eloss_geom.PAR"        

c  Module information:
      CHARACTER*(*)  CRNAME, CRAUTH
      CHARACTER*100  CRMESS
      PARAMETER (CRNAME='CCYLINDER')
      PARAMETER (CRAUTH='Eugene Pasyuk')
c
c  Local User defined variables:
c  -----------------------------
      INTEGER NWRITE
      INTEGER ierr,ierr1,ierr2
      INTEGER i
      REAL point(3)
      REAL pos1(3), pos2(3)
      REAL intersec(3),intersec2(3)
      REAL r
      REAL sind, cosd, tand, angle
      DATA CRCSID/   
     1'$Id: ccylinder.F,v 1.7 2004/12/10 20:15:30 baltzell Exp $'   
     2/
      DATA rayon/9.4/
      DATA thickness/0.2/
      DATA extremtube/17.2/
      DATA alpha/30./
      DATA dir/0., 0., 1./
      DATA pi/3.14159265359/
      DATA NWRITE/0/
      sind(angle) = sin( angle*pi/180. )
      cosd(angle) = cos( angle*pi/180. )
      tand(angle) = tan( angle*pi/180. )
#ifdef Linux

c for Linux these functions are not implemeted in standard library
c we define them here
c
      sind(angle) = sin( angle*pi/180. )
      cosd(angle) = cos( angle*pi/180. )
      tand(angle) = tan( angle*pi/180. )
#endif




c_end_var
c
c  executable code for routine CCYLINDER:
c----6----------------------------------------------------------------72
      ierr1 = 0
      ierr2 = 0
c------------------------------------------------------------------------------
      do i = 1,3
         point(i) = vertex(i) + 50.*cdir(i)
         centre(i) = target_offset(i)
      enddo
c---- 6----------------------------------------------------------------72
c     
      IF (NWRITE .LT. 1) THEN
         NWRITE = NWRITE + 1
         write(CRMESS,*) 
     + 'CCYLINDER: target position is: ', target_offset
c         CRMESS = 'First call to TARGCELL'
c         CALL RECMES(CRNAME,'I',CRMESS)   !commented this one out DK
         
      ENDIF
c     
c------------------------------------------------------------------------------

      call InterCylinder(vertex, cdir, centre, dir, rayon,
     $     pos1, ierr)
      if (ierr .eq. 0 .or. ierr .eq. 2) then

c        No further treatment if:
c         - particle travels along the cylinder axis (ierr = 0)
c         - particle starts outside of the cylinder and travels 
c           away from the cylinder (ierr = 0)
c         - particle starts outside of the cylinder and travels
c           towards the cylinder and would have two intersections (ierr = 2)
c           [ should not happen in the case of a true event ]

         dist = 0.              ! get out
         return
      endif

c--- cylinder or cone?
      if (pos1(3) .gt. extremtube+centre(3) .or. ierr .eq. -1) then

         centre(3) = centre(3)+ extremtube + rayon/tand(alpha)

         call IntersectionCone(vertex, point, centre, dir, alpha,
     +        pos1, pos2, ierr1)

         intersec(1) = pos1(1)
         intersec(2) = pos1(2)
         intersec(3) = pos1(3)
         centre(3) = centre(3) + thickness/sind(alpha)

         call IntersectionCone(vertex, point, centre, dir, alpha,
     +        pos1, pos2, ierr2)

      else
c--- track goes through the cylinder
         intersec(1) = pos1(1)
         intersec(2) = pos1(2)
         intersec(3) = pos1(3)
         r = rayon + thickness

         call InterCylinder(vertex, cdir, centre, dir,
     +        r, pos1, ierr)

      endif

      dist = 0.
      do i = 1,3
         dist = dist + (pos1(i) - intersec(i)) * (pos1(i) - intersec(i))
      enddo

      dist = sqrt(dist)


c   If IntersectionCone above returns a -2, then we're near a singularity
c   and the following code recomputes the intersections, but displaces
c   the track's Theta slightly to either side and averages.

      if(cdir(3).gt.0) then

         if(ierr1.eq.-2.or.ierr2.eq.-2) then

c  First Intersection
           centre(3) = target_offset(3) + extremtube + rayon/tand(alpha)

           point(3) = point(3) + 0.001  ! shift z forward

           call IntersectionCone(vertex, point, centre, dir, alpha,
     +           pos1, pos2, ierr)

           intersec(1) = pos1(1)
           intersec(2) = pos1(2)
           intersec(3) = pos1(3)

           point(3) = point(3) - 0.002  ! shift z backwards an equal amount

           call IntersectionCone(vertex, point, centre, dir, alpha,
     +           pos1, pos2, ierr)
c  average around the singularity
           intersec(1) = (intersec(1)+pos1(1))/2
           intersec(2) = (intersec(2)+pos1(2))/2
           intersec(3) = (intersec(3)+pos1(3))/2

c  Second Intersection
           centre(3) = centre(3) + thickness/sind(alpha)

           point(3) = point(3) + 0.002  ! shift z forward

           call IntersectionCone(vertex, point, centre, dir, alpha,
     +           pos1, pos2, ierr)

           intersec2(1) = pos1(1)
           intersec2(2) = pos1(2)
           intersec2(3) = pos1(3)

           point(3) = point(3) - 0.002  ! shift z backwards an equal amount

           call IntersectionCone(vertex, point, centre, dir, alpha,
     +           pos1, pos2, ierr)
c  average around the singularity
           intersec2(1) = (intersec2(1)+pos1(1))/2
           intersec2(2) = (intersec2(2)+pos1(2))/2
           intersec2(3) = (intersec2(3)+pos1(3))/2
 
           dist = 0.

           do i = 1,3
              dist = dist + ( intersec(i) - intersec2(i) )**2
           enddo

           dist = sqrt(dist)
           
        endif
        
      endif


      RETURN
      END



