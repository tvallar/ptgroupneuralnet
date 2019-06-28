      SUBROUTINE TARGCELL( icell, vertex, cdir, dist, dist1)
c
c_begin_doc
c  RCS ID string
c  $Id: targcell.F,v 1.22 2008/05/07 15:57:06 pmatt Exp $
c
c  Documentation for subroutine TARGCELL
c
c  Purpose: calculates track length within LH2 target and its mylar walls.
c  --------
c           g1 run target geometry
c
c  Input Parameters:  (Name - Type - Meaning)
c  ----------------
c  icell     (integer) target cell type: 
c
c		       icell = 0 - no target
c                      icell = 1 - G1A/G1B/G6A/G6b running period 
c                      icell = 2 - G2A running period 
c                      icell = 3 - G1C running period 
c                      icell = 4 - G3 running period
c                      icell = 5 - G6C/G8A running period 
c                      icell = 6 - g10a running period
c                      icell = 7 - g11a/G13 running period
c                      icell = 8 - EG3a running period
c     
c                
c  vertex(3) (real) vertex coordinates
c  cdir(3)   (real) track direction cosines  
c
c  Output Parameters:  (Name - Type - Meaning)
c  -----------------
c  dist (real)  track length within target material
c  dist1 (real) track length in target wall
c
c  Other routines:
c  ---------------
c  IntersectionSphere, IntersectionCylindre
c
c  Notes:
c  ------
c
c  Author:   Eugene Pasyuk      Created:  Tue Apr  6 16:37:59 EDT 1999
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
c
      INTEGER icell
      REAL vertex(3)
      REAL cdir(3)
      REAL dist
      REAL dist1 
      REAL dir(3)
      REAL LR
      REAL LRC

c     Local pre-defined variables:
c     ---------------------------
c
c      REAL centre(3)

      REAL LC(8)                !Half-length of a cylinder or cone 
      REAL LS(8)                !sphere center offset
      REAL RC(8)                !Cylinder radius ! larger  side radius
      REAL RS(8)                !Sphere radius
      REAL WALL(8)              !wall thickness
      
      REAL RCON(3)              ! smaller radius of cone 
      REAL ALPHA(3)             ! cone angle (half angle)

c     RCS information: 

      CHARACTER*132  CFILE, CREVIS, CSTATE, CDATE, CAUTHO, CRCSID
      PARAMETER (CFILE = '$RCSfile: targcell.F,v $')
      PARAMETER (CREVIS = '$Revision: 1.22 $')
      PARAMETER (CSTATE = '$State: Exp $')
      PARAMETER (CDATE = '$Date: 2008/05/07 15:57:06 $')
      PARAMETER (CAUTHO = '$Author: pmatt $')
      DATA CRCSID/   
     1     '$Id: targcell.F,v 1.22 2008/05/07 15:57:06 pmatt Exp $'/
      
c     Module information:

      CHARACTER*(*)  CRNAME, CRAUTH
      CHARACTER*100  CRMESS
      PARAMETER (CRNAME='TARGCELL')
      PARAMETER (CRAUTH='Eugene Pasyuk')
c     
c     Local User defined variables:
c     -----------------------------
c
      INTEGER NWRITE
      INTEGER ierr
      INTEGER ierr1
      INTEGER ierr2
c     ABOVE WAS ADDED!!
      INTEGER extrm
      INTEGER i

      REAL pos1(3)
      REAL pos2(3)
      REAL pos3(3)
      REAL pos4(3)
      REAL pos1w(3)
      REAL pos2w(3)
      REAL pos3w(3)
      REAL pos4w(3)
      REAL point(3)
      REAL lastwall
      REAL vc(3)
      REAL cc(3)
      REAL r,r1,r2,h
      REAL x
      REAL pi
      REAL sind, cosd, tand, angle
#include "eloss_geom.PAR"

      DATA pi/3.14159265359/
      DATA nwrite/0/
      
      DATA LC/6.0, 4.5, 8.5, 8.5, 8.5  ,11.15, 19.15,  19.15/ !20.0
      DATA LS/6.0, 1.0, 5.0, 5.0, 6.85 , 8.0 , 16.0 ,  16.0 / !
      DATA RC/3.0, 2.0, 2.0, 2.0, 1.4  , 0.0 ,  2.0 ,  2.0  / !
      DATA RS/3.0, 4.0, 4.0, 4.0, 2.248, 4.0 ,  4.0 ,  4.0  / !

c wall thickness includes superinsulation, which is about 15 mg/cm^2
ccc      DATA WALL/0.017, 0.0127, 0.0127, 0.019/ !originall wall thickness 
      
      DATA WALL/0.032,0.0277,0.0277,0.034,0.0277,0.0277,0.0227,0.0227/ 


      DATA ALPHA/1.7 , 0.87, 0.87/
      DATA RCON /2.0 , 2.0, 2.0 /

      DATA dir /0., 0., 1./     !target axis direction cosines
      
      
      
      
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


c     _end_var
c     
c     executable code for routine LH2TARG:
c---- 6----------------------------------------------------------------72
      
      dist = 0.
      dist1 = 0.

      if( icell .LE. 0 .OR. icell .GT. 8 ) then
        write(CRMESS,*) 
     +           'TARGCELL:  unknown cell ',icell
C            CALL RECMES(CRNAME,'F',CRMESS)
	return
      endif



C--- cylindrical cells for runs before g10
      if(icell .le. 5) then 
         do i = 1,3
            point(i) = vertex(i) + 1000.*cdir(i)
            cc(i) = target_offset(i)
         enddo
c---- 6----------------------------------------------------------------72
c     
         IF (NWRITE .LT. 1) THEN
            NWRITE = NWRITE + 1
            write(CRMESS,*) 
     +           'TARGCELL: target position is: ', target_offset
c            CALL RECMES(CRNAME,'I',CRMESS)
         ENDIF
c     
c------------------------------------------------------------------------------
         
         call InterCylinder(vertex,cdir,cc,dir,RC(icell),pos1,ierr)

         if (ierr .eq. 0 .or. ierr .eq. 2 ) then
            dist = 0.
            dist1 = 0.
            return              ! no intersection, get out
         endif
         
c---  cylinder or sphere?
         extrm = 0
         if ( pos1(3) .gt. LC(icell)+target_offset(3) ) extrm = 1
         if ( target_offset(3)-LC(icell) .gt. pos1(3)) extrm = -1
         
c---  sphere
         if (extrm .ne. 0) then 
            cc(3) = extrm*LS(icell) + target_offset(3)
            call IntersectionSphere(vertex, point, cc, RS(icell),
     $           pos1, pos2, ierr)
            if (ierr .eq. 0) then 
               dist = 0.
               dist1 = 0. 
               return           !no intersection, get out
            endif
c---  spheric wall
            r = RS(icell) + wall(icell)
            call IntersectionSphere(vertex, point, cc, r,
     $           pos1w, pos2w, ierr1)
            
            if (ierr .eq. 2) then
               pos1(1) = pos2(1)
               pos1(2) = pos2(2)
               pos1(3) = pos2(3)
               pos1w(1) = pos2w(1)
               pos1w(2) = pos2w(2)
               pos1w(3) = pos2w(3)
            endif
c---  cylinder wall
         else 
            r = RC(icell) + wall(icell)
            call InterCylinder(vertex, cdir, cc, dir, r,
     $           pos1w, ierr)
         endif
         
         dist = 0.
         x = 0.
         
         do i = 1,3
            dist = dist + (pos1(i) - vertex(i))*(pos1(i) - vertex(i))
            x = x + (pos1w(i) - vertex(i))*(pos1w(i) - vertex(i))
         enddo
         
         dist = sqrt(dist)
         x = sqrt(x)
         dist1 = x - dist
         


      else if( icell .GE. 6 .AND. icell .LE. 8 ) then
c---------------------------------------------------------------------------
c     -- g10a cell (6) or g11a cell (7) or eg3 cell (8) --

         do i = 1,3
            point(i) = vertex(i) + 1000.*cdir(i)
            cc(i) = target_offset(i)
         enddo

ccc   - first make sure vertex is in the target
         if(abs(vertex(3)-cc(3)) .le. LS(icell)+RS(icell)) then
c     -     vertex Z is within the target
            vc(1) = cc(1)
            vc(2) = cc(2)
         vc(3) = cc(3)+LC(icell)+RCON(icell-5)/tand(alpha(icell-5))            
         call IntersectionCone(vertex,point,vc,dir,alpha(icell-5),
     +                    pos1, pos2, ierr)

            dist1 = 0.
            dist = 0.
            x = 0.
            if((ierr .eq. 1 .or. ierr .eq. 2)) then
c              There is at least one valid intercept and it's in the correct direction
               vc(3) = vc(3) + wall(icell)/tand(alpha(icell-5))
            call IntersectionCone(vertex,point,vc,dir,alpha(icell-5),
     +              pos1w, pos2w, ierr2)
               if(pos1(3) .lt. cc(3)+LC(icell) .and. ierr .eq. 1) then
c                 it's a one hit cone only
                  do i = 1,3
                     dist = dist+(pos1(i)-vertex(i))*(pos1(i)-vertex(i))
                     x = x + (pos1w(i)-vertex(i))*(pos1w(i)-vertex(i))
                  enddo
                  dist = sqrt(dist)
                  x = sqrt(x)
                  dist1 = x - dist
           else if(pos1(3) .lt. cc(3)+LC(icell) .and. pos2(3) .lt.
     +     cc(3)+LC(icell) .and. ierr .eq. 2) then
c                 it's a two hit cone
                  do i = 1,3
                     dist = dist+(pos1(i)-pos2(i))*(pos1(i)-pos2(i))
                     x = x + (pos1w(i)-pos2w(i))*(pos1w(i)-pos2w(i))
                  enddo
                  dist = sqrt(dist)
                  x = sqrt(x)
                  dist1 = x - dist
               else
c                 it's a sphere
                  extrm=1
                  cc(3) = extrm*LS(icell) + target_offset(3)
                  call IntersectionSphere(vertex, point,cc,RS(icell),
     $                 pos3, pos4, ierr2)
                  if (ierr2 .eq. 0) then 
                     return     !no intersection, get out
                  endif
c---  spheric wall
                  r = RS(icell) + wall(icell)
                  call IntersectionSphere(vertex, point, cc, r,
     $                 pos3w, pos4w, ierr1)
                  if (ierr2 .eq. 2) then
                     pos3(1) = pos4(1)
                     pos3(2) = pos4(2)
                     pos3(3) = pos4(3)
                     pos3w(1) = pos4w(1)
                     pos3w(2) = pos4w(2)
                     pos3w(3) = pos4w(3)
                  endif
                  if(ierr .ne. 2) then
c                    sphere only
                     do i = 1,3
                     dist = dist+(pos3(i)-vertex(i))*(pos3(i)-vertex(i))
                     dist1 = dist1+(pos3w(i)-pos3(i))*(pos3w(i)-pos3(i))
                     enddo
                     dist = sqrt(dist)
                     dist1 = sqrt(dist1)
c--- account for aluminum endcap
                     if(pos3w(1)**2 + pos3w(2)**2 .lt. 0.5**2.and. 
     +                  cdir(3).gt.0  ) then
                        dist1 = dist1/wall(icell) *
     +                     ( wall(icell)
     +                     +0.0015*2.70/1.032) ! Aluminum wall    (rho = 2.70 g/cm^2)
                     endif
              else
c             it hit the sphere and the cone -> pos1 and pos3
              do i = 1,3
              dist = dist + (pos1(i)-pos3(i))*(pos1(i)-pos3(i))
              lastwall=lastwall+(pos3w(i)-pos3(i))*(pos3w(i)- pos3(i))
              dist1 = dist1 + (pos1w(i) - pos1(i))*(pos1w(i) - pos1(i))
              enddo
              lastwall = sqrt(lastwall)
              dist = sqrt(dist)
              dist1 = lastwall + sqrt(dist1)
c--- account for aluminum endcap
              if(pos3w(1)**2 + pos3w(2)**2 .lt. 0.5**2.and. 
     +        cdir(3).gt.0  ) then
              dist1 = dist1 + lastwall*0.0015*2.70/(1.032*wall(icell)) ! Aluminum wall    (rho = 2.70 g/cm^2)
              endif
              endif
              endif
            endif
         endif

      endif

      
      RETURN
      END


