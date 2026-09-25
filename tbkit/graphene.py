from __future__ import annotations

from tbkit.lattice import *
from tbkit.plot import *
from tbkit.system import *
from math import sqrt

PI = np.pi
ATOL = 1e-3
DX = 0.5 * sqrt(3)
DY = 0.5

 
#################################
# CLASS GRAPHENE
#################################

       
class GrapheneLattice(Lattice):
    def __init__(self) -> None:
        unit_cell = [{'tag': 'a', 'r0': (0, 0)},
                          {'tag': 'b', 'r0': (DX, DY)}]
        prim_vec = [(2*DX, 0.), (DX, 1.5)]
        Lattice.__init__(self, unit_cell=unit_cell, prim_vec=prim_vec)
        self.butterfly = np.array([])
        self.betas = np.array([])

    def triangle_zigzag(self, n: int) -> None:
        '''
        Triangular flake with zigzag terminations.

        :param: n. Int. Number of plaquettes along the edges. 
        '''
        error_handling.positive_int(n, 'n')
        self.get_lattice(n1=n+2, n2=n+2)
        self.boundary_line(cx=-sqrt(3), cy=-1, co=-3*n+2)

    def hexagon_zigzag(self, n: int) -> None:
        '''
        Hexagonal flake with zigzag terminations.

        :param: n. Int. Number of plaquettes along the edges. 
        '''
        error_handling.positive_int(n, 'n')
        self.get_lattice(n1=2*n, n2=2*n)
        self.boundary_line(cx=sqrt(3), cy=1, co=3*(n-1))
        self.boundary_line(cx=-sqrt(3), cy=-1, co=-9*n+2.5)

    def triangle_armchair(self, n: int) -> None:
        '''
        Triangular flake with armchair terminations.

        :param: n. Int. Number of plaquettes along the edges. 
        '''
        error_handling.positive_int(n, 'n')
        self.get_lattice(n1=2*n, n2=2*n)
        self.boundary_line(cx=1, cy=0, co=sqrt(3)/2*(2*n-1)-0.1)
        self.boundary_line(cx=-1/sqrt(3), cy=1, co=-n-0.1)
        self.boundary_line(cx=-1/sqrt(3), cy=-1, co=-4*n+0.1)

    def hexagon_armchair(self, n: int) -> None:
        '''
        Hexagonal flake with armchair terminations.

        :param: n. Int. Number of plaquettes along each edge. 
        '''
        error_handling.positive_int(n, 'n')
        nn = 3 * n - 2
        self.get_lattice(n1=2*nn, n2=2*nn)
        self.boundary_line(cx=1, cy=0, co=sqrt(3)/2* (2*nn-1)-.1)
        self.boundary_line(cx=-1/sqrt(3), cy=1, co=-nn -.1)
        self.boundary_line(cx=-1/sqrt(3), cy=-1, co=-4*nn +1-.1)
        self.coor['x'] -= self.coor['x'].min()
        self.boundary_line(cx=-1, cy=0, co=-DX * 2*nn-0.1)
        self.boundary_line(cx=1/sqrt(3), cy=1, co=3*n -3.)
        self.boundary_line(cx=1/sqrt(3), cy=-1, co=-6*n+4)

    def square(self, n: int) -> None:
        '''
        Squared flake.

        :param: n. Int. Number of plaquettes along x. 
        '''
        error_handling.positive_int(n, 'n')
        n2 = int(1.5*DX*n)
        self.get_lattice(n1=2*n, n2=n2)
        self.boundary_line(cx=1, cy=0, co=DX*(2*n-2)+0.1)
        self.boundary_line(cx=-1, cy=0, co=-DX*(4*n)+0.5)

    def circle(self, n: int) -> None:
        '''
        Circular flake.

        :param: n. Int. Number of plaquettes along the diameter. 
        '''
        error_handling.positive_int(n, 'n')
        self.get_lattice(n1=2*n, n2=2*n)
        self.sites = len(self.coor)
        self.center()
        if n % 2 == 0:
            self.shift_x(shift=-DX)
        self.ellipse_in(rx=DX*(n+1), ry=DX*(n+1), x0=0., y0=0.)
        self.remove_dangling()
        
class GrapheneSystem(System):
    def __init__(self, lat: Lattice) -> None:
        System.__init__(self, lat)

    def _strain_projection(
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.float64], NDArray[np.float64]]:
        r'''
        Private method.

        Get, for every nearest-neighbor bond, the projection

        .. math::

            s_{ij} = \hat{\boldsymbol\delta}_{ij}\cdot\mathbf{r}_{ij}

        of its midpoint :math:`\mathbf{r}_{ij}` on its direction
        :math:`\hat{\boldsymbol\delta}_{ij}`, oriented from its 'b' site to
        its 'a' site (the same orientation for all three bond families), so
        that the strained hopping is :math:`t_{ij} = t(1 + \tfrac14\beta s_{ij})`.

        :returns:
            * **i**, **j** -- Integer ndarrays. Bond end points, oriented as in
              *System.get_bonds*.
            * **ang** -- Real ndarray. Bond angles (from *i* to *j*), in degrees,
              in [0, 180).
            * **s** -- Real ndarray. The projections above.
        '''
        self.get_distances()
        i, j, ang = self.get_bonds(1)
        # direction from the 'b' end to the 'a' end of each bond
        ang_ba = np.where(self.lat.coor['tag'][i] == 'a', ang - 180., ang)
        x_center = .5 * (self.lat.coor['x'][i] + self.lat.coor['x'][j])
        y_center = .5 * (self.lat.coor['y'][i] + self.lat.coor['y'][j])
        s = (np.cos(PI / 180 * ang_ba) * x_center + np.sin(PI / 180 * ang_ba) * y_center)
        return i, j, ang, s

    def set_hop_linear_strain(self, t: complex, beta: float) -> None:
        r'''
        Set nearest-neighbor hoppings according to a linear triaxial strain:

        .. math::

            t_{ij} = t\left(1 + \tfrac14\beta\,
                     \hat{\boldsymbol\delta}_{ij}\cdot\mathbf{r}_{ij}\right)

        with :math:`\hat{\boldsymbol\delta}_{ij}` the bond direction (from its
        'b' site to its 'a' site) and
        :math:`\mathbf{r}_{ij}` its midpoint. The strain is measured from the
        coordinate origin, so centre the flake on it (see *lattice.center*)
        before calling this.

        :param t: Hopping value without strain.
        :param beta: Strength of the strain. See *get_beta_lims* for the
            range that keeps every hopping positive.
        '''
        error_handling.number(t, 't')
        error_handling.real_number(beta, 'beta')
        i, j, ang, s = self._strain_projection()
        self.hop = np.zeros(len(i), dtype=HOP_DTYPE)
        self.hop['n'] = 1
        self.hop['i'] = i
        self.hop['j'] = j
        self.hop['ang'] = ang
        self.hop['tag'] = npc.add(self.lat.coor['tag'][i], self.lat.coor['tag'][j])
        self.hop['t'] = t * (1. + 0.25 * beta * s)

    def get_butterfly(self, t: complex, N: int) -> None:
        '''
        Get the spectrum as a function of the strain *beta* (see
        *set_hop_linear_strain*), stored in *sys.butterfly*, shape
        (N, sites), with the strain values in *sys.betas*.

        :param t: Unstrained hopping value.
        :param N: Positive integer. Number of strain values between the
            minimal and maximal strains given by *get_beta_lims*.
        '''
        error_handling.number(t, 't')
        error_handling.positive_int(N, 'N')
        beta_lims = self.get_beta_lims()
        self.betas = np.linspace(beta_lims[0], beta_lims[1], N)
        self.butterfly = np.zeros((N, self.lat.sites))
        for i, beta in enumerate(self.betas):
            self.set_hop_linear_strain(t=t, beta=beta)
            self.get_ham()
            self.butterfly[i] = LA.eigvalsh(self.ham.toarray())

    def get_beta_lims(self) -> NDArray[np.float64]:
        r'''
        Get the extremal strain values keeping every hopping positive.

        Under *set_hop_linear_strain* a bond's amplitude is
        :math:`t(1+\tfrac14\beta s_{ij})`, so it stays positive for every bond
        iff :math:`-4/\max_{ij} s_{ij} < \beta < -4/\min_{ij} s_{ij}`.

        :returns:
            * **beta_lims** -- Real ndarray of length 2, ``[beta_min, beta_max]``
              (ascending). A bound is infinite if the corresponding
              :math:`s_{ij}` never takes that sign.
        '''
        _, _, _, s = self._strain_projection()
        eps = 1e-6
        s_min, s_max = s.min(), s.max()
        beta_min = -4. / s_max + eps if s_max > 0 else -np.inf
        beta_max = -4. / s_min - eps if s_min < 0 else np.inf
        return np.array([beta_min, beta_max])


# Backward-compatible camelCase aliases (pre-0.2 API).
grapheneLat = GrapheneLattice
grapheneSys = GrapheneSystem
