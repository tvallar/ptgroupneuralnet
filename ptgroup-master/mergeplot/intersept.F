C
C set of subroutines which find an intersection, if any, of a strait line
C with a cylinder, plane, sphere and cone
c

      subroutine InterCylinder (vertex, cdir, center, d, R, x, ierr)

      real vertex(3), cdir(3), center(3)
      real d(3), r
      real x(3)
      real vp(3), va(3), vb(3)
      real a, b, c
      real k1, k2, k

      real vp_dot_d
      real c_dot_d

      integer i

      ierr = -1

      do i = 1, 3
         vp(i) = vertex(i) - center(i)
         x(i) = 0.
      enddo

      vp_dot_d = 0.
      c_dot_d = 0.
      do i = 1, 3
         vp_dot_d = vp_dot_d + vp(i) * d(i)
         c_dot_d = c_dot_d + cdir(i) * d(i)
      enddo

      do i = 1, 3
         va(i) = cdir(i) - c_dot_d * d(i)
         vb(i) = vp(i) - vp_dot_d * d(i)
      enddo

      a = 0.
      b = 0.
      c = 0.
      do i = 1, 3
         a = a + va(i) * va(i)
         b = b + 2. * va(i) * vb(i)
         c = c + vb(i) * vb(i)
      enddo
      c = c - R * R

      delta = b * b - 4 * a * c

      if (delta .ge. 0.) then
         k1 = (-b + sqrt(delta)) / a / 2
         k2 = (-b - sqrt(delta)) / a / 2
         if (k1 .ge. 0.) then

c           valid intercept

            if (k2 .gt. 0.) then

c              two intercepts in the direction of the track

               k = k2
               ierr = 2

            else

c              one and only one intercept in the direction of the track

               k = k1
               ierr = 1
            
            endif

            do i = 1, 3
               x(i) = vertex(i) + k * cdir(i)
            enddo

         else

c           particle passed the cylinder already.            
c           no more intercept in the direction of the track.

            ierr = 0
         endif

      else

c        no intercept with cylinder

         ierr = 0
      endif
      end


c**********************************************************************

      subroutine IntersectionCylindre(m1, m2, m0, d, r, m, mm, rep)

c**********************************************************************


      implicit NONE

      real m1(3), m2(3), m0(3), d(3), m(3), mm(3), r
      real mm1(4), mm2(4), mm0(4), dd(4)
      integer rep, i
      real a, b, c, k1, k2, kt, delta, norme

      norme = 0.
      do i = 1,3
         norme = norme + d(i)*d(i)
      enddo
      
      if (1e-10 .gt. norme) then 
         rep = -3
         return
      endif

      do i = 1,3
         d(i) = d(i)/norme
      enddo

      do i = 1,3
         mm1(i) = m1(i)
         mm2(i) = m2(i)
         mm0(i) = m0(i)
         dd(i) = d(i)
      enddo

      mm1(4) = m1(1)
      mm2(4) = m2(1)
      mm0(4) = m0(1)
      dd(4) = d(1)

      a = 0.
      do i = 1,3
         a = a + (m2(i) - m1(i)) * (m2(i) - m1(i)) * (1. - d(i)*d(i))
     +        -2.*(m2(i) - m1(i)) * (mm2(i+1) - mm1(i+1)) *d(i)*dd(i+1)
      enddo
      
      if (1e-10 .gt. abs(a)) then
         rep = 0
         return
      endif

      b = 0.
      do i = 1,3
         b = b + (m2(i) - m1(i))*(m1(i) - m0(i))*(1. - d(i)*d(i))
         b = b - (m2(i) - m1(i))*(mm1(i+1) - mm0(i+1))*d(i)*dd(i+1)
         b = b - (m1(i) - m0(i))*(mm2(i+1) - mm1(i+1))*d(i)*dd(i+1)
      enddo

      c = 0.
      do i = 1,3
         c = c + (m1(i) - m0(i))*(m1(i) - m0(i))*(1. - d(i)*d(i))
         c = c - 2.*(m1(i) - m0(i))*(mm1(i+1) - mm0(i+1))*d(i)*dd(i+1)
      enddo
      c = c - r*r     
      delta = b*b - a*c
      if (0 .gt. delta) then
         rep = 0
         return
      endif

      k1 = (-b + sqrt(delta))/a
      k2 = (-b - sqrt(delta))/a

      rep = -1
      if (k1 .gt. 0. .and. 1. .gt. k1) rep = 1
      if (k2 .gt. 0. .and. 1. .gt. k2) then
         if (rep .eq. 1) then
            rep = 2
         else
            rep = 1
            kt = k1
            k1 = k2
            k2 = kt
         endif
      endif


      if (rep .eq. 2 .or. rep .eq. -1) then
         if (abs(k1) .gt. abs(k2)) then
            kt = k1
            k1 = k2
            k2 = kt
         endif
      endif
      
      do i = 1,3
         m(i) = m1(i) + k1*(m2(i) - m1(i))
         mm(i) = m1(i) + k2*(m2(i) - m1(i))
      enddo

      return
      end	
      
c     ******************************************************************

      subroutine IntersectionPlan(m1, m2, m0, n, m, rep)

c     ******************************************************************


      implicit NONE

      real m1(3), m2(3), m0(3), n(3), m(3)
      integer rep, i
      real kd, k

      if (n(1) .eq. 0. .and. n(2) .eq. 0. .and. n(3) .eq. 0.) then
         rep = -3
         return
      endif	

      rep = 0
      k = 0.
      do i = 1,3 
         k = k + (m1(i) - m0(i))*n(i)
      enddo
      
      if (1e-10 .gt. abs(k)) rep = -2

      kd = 0.
      do i = 1,3
         kd = kd + (m2(i) - m1(i))*n(i)
      enddo

      if (1e-10 .gt. kd) then
         if (rep .ne. -2) rep = 0
         return
      endif

      k = 0.
      do i = 1,3
         k = k + (m0(i) - m1(i))*n(i)
      enddo

      k = k/kd
      if (k .gt. 0. .and. 1. .gt. k)	then 
         rep = 1 
      else 
         rep = -1
      endif

      do i = 1,3
         m(i) = m1(i) + k*(m2(i) - m1(i))
      enddo

      return
      end	
      
c     ******************************************************************

      subroutine IntersectionSphere(m1, m2, m0, r, m, mm, rep)

c     ******************************************************************

      implicit NONE

      real m1(3), m2(3), m0(3), m(3), mm(3), r
      integer rep, i
      real a, b, c, k1, k2, kt, delta

      a = 0.
      do i = 1,3
         a = a + (m2(i) - m1(i))*(m2(i) - m1(i))
      enddo

      if (1e-10 .gt. abs(a)) then
         rep = 0
         return
      endif

      b = 0.
      do i = 1,3
         b = b + (m2(i) - m1(i))*(m1(i) - m0(i))
      enddo

      c = 0.
      do i = 1,3
         c = c + (m1(i) - m0(i))*(m1(i) - m0(i))
      enddo

      c = c - r*r
      delta = b*b - a*c
      if (0.gt.delta) then
         rep = 0
         return
      endif

      k1 = (-b + sqrt(delta))/a
      k2 = (-b - sqrt(delta))/a

      rep = -1
      if (k1 .gt. 0. .and. 1. .gt. k1) rep = 1
      if (k2 .gt. 0. .and. 1. .gt. k2) then
         if (rep .eq. 1) then
            rep = 2
         else
            rep = 1
            kt = k1
            k1 = k2
            k2 = kt
         endif
      endif

      if (rep .eq. 2 .or. rep .eq. -1) then
         if (abs(k1) .gt. abs(k2)) then
            kt = k1
            k1 = k2
            k2 = kt
         endif
      endif
      
      do i = 1,3
         m(i) = m1(i) + k1*(m2(i) - m1(i))
         mm(i) = m1(i) + k2*(m2(i) - m1(i))
      enddo

      return
      end	

c     ******************************************************************

      subroutine IntersectionCone(m1, m2, m0, d, alpha, m, mm, rep)

c     ******************************************************************

      implicit NONE

      real m1(3), m2(3), m0(3), d(3), m(3), mm(3), alpha
      real mm1(4), mm2(4), mm0(4), dd(4)
      integer rep, i
      real a, b, c, k1, k2, kt, delta, norme, c2, a1, a2

      c2 = cos(alpha/180.*3.14159265359)
      c2 = c2*c2

      norme = 0.
      do i = 1,3
         norme = norme + d(i)*d(i)
      enddo
      
      if (1e-10 .gt. norme) then 
         rep = -3
         return
      endif

      do i = 1,3
         d(i) = d(i)/norme
      enddo

      do i = 1,3
         mm1(i) = m1(i)
         mm2(i) = m2(i)
         mm0(i) = m0(i)
         dd(i) = d(i)
      enddo

      mm1(4) = m1(1)
      mm2(4) = m2(1)
      mm0(4) = m0(1)
      dd(4) = d(1)

      a = 0.
      do i = 1,3

         a = a + (m2(i) - m1(i))*(m2(i) - m1(i))*(d(i)*d(i) - c2)
     +        +2.*(m2(i) - m1(i))*(mm2(i+1) - mm1(i+1))*d(i)*dd(i+1)
      enddo
      
      if (1e-10 .gt. abs(a)) then
         rep = 0
         return
      endif

      b = 0.
      do i = 1,3
         b = b + (m2(i) - m1(i))*(m1(i) - m0(i))*(d(i)*d(i) - c2)
         b = b + (m2(i) - m1(i))*(mm1(i+1) - mm0(i+1))*d(i)*dd(i+1)
         b = b + (m1(i) - m0(i))*(mm2(i+1) - mm1(i+1))*d(i)*dd(i+1)
      enddo

      c = 0.
      do i = 1,3
         c = c + (m1(i) - m0(i))*(m1(i) - m0(i))*(d(i)*d(i) - c2)
         c = c + 2.*(m1(i) - m0(i))*(mm1(i+1) - mm0(i+1))*d(i)*dd(i+1)
      enddo
      
      delta = b*b - a*c
      if (0. .gt. delta) then
         rep = 0
         return
      endif

      a1 =  -b + sqrt(delta)
      a2 =  -b - sqrt(delta)
      k1 = a1 / a
      k2 = a2 / a

      rep = -1
      if (k1 .gt. 0. .and. 1. .gt. k1) rep = 1
      if (k2 .gt. 0. .and. 1. .gt. k2) then
         if (rep .eq. 1) then
            rep = 2
         else
            rep = 1
            kt = k1
            k1 = k2
            k2 = kt
            kt = a1
            a1 = a2
            a2 = kt
         endif
      endif

      if (rep .eq. 2 .or. rep .eq. -1) then
         if (abs(k1) .gt. abs(k2)) then
            kt = k1
            k1 = k2
            k2 = kt
            kt = a1
            a1 = a2
            a2 = kt
         endif
      endif
      
c Check if k = a1/a is near a singularity and return -2 flag
      if( abs(a1).lt.1e-4 .and. abs(a).lt.1e-3 ) rep = -2

      do i = 1,3
         m(i)  = m1(i) + k1*(m2(i) - m1(i))
         mm(i) = m1(i) + k2*(m2(i) - m1(i))
      enddo

      return
      end	
      










