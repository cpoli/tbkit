import numpy as np
import inspect


ATOL = 1e-3  # matches the distance/angle tolerance used throughout tbkit.system


###############################
# GENERIC EXCEPTION HANDLING
###############################


def boolean(var, var_name):
    '''
    Check if *var* is a boolean.

    :raises TypeError: Parameter *var* must be a bool.
    '''
    if not isinstance(var, bool):
        raise TypeError('\n\nParameter {} must be a bool.\n'.format(var_name))


def positive_int(var, var_name):
    '''
    Check if *var* is a positive integer.

    :raises TypeError: Parameter *var* must be an integer.
    :raises ValueError: Parameter *var* must be a positive integer.
    '''
    if not isinstance(var, int):
        raise TypeError('\n\nParameter {} must be an integer.\n'.format(var_name))
    if var < 1:
        raise ValueError('\n\nParameter {} must be a positive integer.\n'.format(var_name))


def positive_int_lim(var, var_name, nmax):
    '''
    Check if *var* is a positive integer smaller than nmax.

    :raises TypeError: Parameter *var* must be an integer.
    :raises ValueError: Parameter *var* must be a positive integer.
    :raises ValueError: Parameter *var* must be a positive integer
      smaller than nmax.
    '''
    if not isinstance(var, int):
        raise TypeError('\n\nParameter {} must be an integer.\n'.format(var_name))
    if var < 1:
        raise ValueError('\n\nParameter {} must be a positive integer.\n'.format(var_name))
    if var > nmax:
        raise ValueError('\n\nParameter {} must be a positive integer '
                                  'smaller than {}.\n'.format(var_name, nmax))


def real_number(var, var_name):
    '''
    Check if parameter *var* is a real number.

    :raises TypeError: Parameter *var* must be a real number.
    '''         
    if not isinstance(var, (int, float)):
        raise TypeError('\n\nParameter {} must be a real number.\n'.format(var_name))


def positive_real(var, var_name):
    '''
    Check if parameter *var* is a positive number.

    :raises TypeError: Parameter *var* must be a real number.
    :raises ValueError: Parameter *var* must be a positive number.
    '''         
    if not isinstance(var, (int, float)):
        raise TypeError('\n\nParameter {} must be a real number.\n'.format(var_name))
    if var <= 0:
        raise ValueError('\n\nParameter {} must be a positive number.\n'.format(var_name))


def positive_real_zero(var, var_name):
    '''
    Check if parameter *var* is a positive number or zero.

    :raises TypeError: Parameter *var* must be a real number.
    :raises ValueError: Parameter *var* must be a positive number or zero.
    '''         
    if not isinstance(var, (int, float)):
        raise TypeError('\n\nParameter {} must be a real number.\n'.format(var_name))
    if var < 0:
        raise ValueError('\n\nParameter {} must be a positive number or zero.\n'.format(var_name))


def negative_real(var, var_name):
    '''
    Check if parameter *var* is a negative number.

    :raises TypeError: Parameter *var* must be a real number.
    :raises ValueError: Parameter *var* must be a negative number.

    '''         
    if not isinstance(var, (int, float)):
        raise TypeError('\n\nParameter {} must be a real number.\n'.format(var_name))
    if var >= 0:
        raise ValueError('\n\nParameter {} must be a negative number.\n'.format(var_name))


def number(var, var_name):
    '''
    Check if parameter *var* is a number.

    :raises TypeError: Parameter *var* must be a real or complex number.
    '''
    if not isinstance(var, (int, float, complex)):
        raise TypeError('\n\nParameter {} must be a real or complex number.\n'.format(var_name))


def is_callable(var, var_name):
    '''
    Check if parameter *var* is callable.

    :raises TypeError: Parameter *var* must be callable.
    '''
    if not callable(var):
        raise TypeError('\n\nParameter {} must be callable.\n'.format(var_name))


def larger(var1, var_name1, var2, var_name2):
    '''
    Check if *var1* larger than *val*.

    :raises ValueError: Parameter *var1* larger than *var2*.
    '''         
    if var1 >= var2:
        raise ValueError('\n\n{} must be larger than {}.\n'
                                    .format(var_name1, var_name2))


def smaller(var1, var_name1, var2, var_name2):
    '''
    Check if *var1* smaller than *var2*.

    :raises ValueError: Parameter *var1* must be smaller than *var2*.
    '''         
    if var1 >= var2:
        raise ValueError('\n\n{} must be smaller than {}.\n'
                                    .format(var_name1, var_name2))


def string(var, var_name):
    '''
    Check if parameter *var* is a string.

    :raises TypeError: Parameter *var* must be a string.
    '''
    if var is None:
        return
    if not isinstance(var, str):
        raise TypeError('\n\nParameter {} must be a string.\n'.format(var_name))


def ndarray(var, var_name, length):
    '''
    Check if parameter *var* is a numpy array.

    :raises TypeError: Parameter *var* must be a numpy ndarray.
    :raises ValueError: length array must be equal to length.
    '''
    if not isinstance(var, np.ndarray):
        raise TypeError('\n\nParameter {} must be a numpy ndarray.\n'.format(var_name))
    if len(var) != length:
        raise ValueError('\n\nParameter {} must be a numpy ndarray of {}\n'
                                    ''.format(var_name, length))


def ndarray_null(var, var_name):
    '''
    Check if parameter *var* is not a null numpy array.

    :raises ValueError: Parameter *var* must not be a null numpy ndarray.
    '''
    array_null = np.zeros(len(var))
    if np.allclose(var, array_null):
        raise ValueError('\n\nParameter {} must not be a null numpy ndarray.\n'.format(var_name))


def ndarray_empty(var, var_name):
    '''
    Check if parameter *var* is not an empty numpy array.

    :raises ValueError: Parameter *var* must not be an emptynumpy ndarray.
    '''
    if var.size == 0:
        raise ValueError('\n\nParameter {} must not be an empty numpy ndarray.\n'.format(var_name))


def list_tuple_2elem(var, var_name):
    '''
    Check if parameter *var* is a list/tuple with 2 elements.

    :raises TypeError: Parameter *var* must be a list/tuple.
    :raises ValueError: Parameter *var* must contain 2 elements.
    '''
    if var is None:
        return
    if not isinstance(var, (list, tuple)):
        raise TypeError('\n\nParameter {} must be a list/tuple.\n'.format(var_name))
    if len(var) != 2:
        raise ValueError('\n\nParameter {} must be a list/tuple of length two.\n'.format(var_name))


def tuple_2elem(var, var_name):
    '''
    Check if parameter *var* is a tuple with 2 elements.

    :raises TypeError: Parameter *var* must be a tuple.
    :raises ValueError: Parameter *var* must contain 2 elements.
    '''
    if var is None:
        return
    if not isinstance(var, tuple):
        raise TypeError('\n\nParameter {} must be a tuple.\n'.format(var_name))
    if len(var) != 2:
        raise ValueError('\n\nParameter {} must be a list/tuple of length two.\n'.format(var_name))


###############################
# LATTICE EXCEPTION HANDLING
###############################


def lat(lat):
    '''
    Check if parameter is an instance of the *lattice*.
    :raises TypeError: Parameter must be an instance of the class lattice.
    '''
    names = inspect.getmro(lat.__class__)
    if str(names).find('lattice') == -1:
        raise TypeError('\n\nParameter must be an instance of the class lattice.\n')


def unit_cell(unit_cell):
    '''
    Check parameter *unit_cell*.

    :raises TypeError: Parameter unit_cell must be a list.
    :raises KeyError: Dictionaries must contain the key "tag".
    :raises KeyError: Dictionaries must contain the key "r0".
    :raises TypeError: Key "tags" must contain a one-character string.
    :raises ValueError: Key "tags" must contain a one-character string.
    :raises ValueError: Key "r0" must contain be a list.
    :raises TypeError: Key "r0" must contain be a tuple.
    :raises ValueError: Key "r0" must contain a tuple of length two.
    :raises ValueError: Key "r0" must contain a tuple of two real numbers.
    '''
    if not isinstance(unit_cell, list):
        raise TypeError('\n\nParameter unit_cell must be a list.\n')
    for dic in unit_cell:
        if 'tag' not in dic:
            raise KeyError('\n\nDictionaries must contain the key "tag".\n')
        if 'r0' not in dic:
            raise KeyError('\n\nDictionaries must contain the key "r0".\n')
        if not isinstance(dic['tag'], str):
            raise TypeError('\n\nKey "tag" must contain a one-character string.\n')
        if not len(dic['tag']) == 1:
            raise ValueError('\n\nKey "tag" must be a one-character string.\n')
        if not isinstance(dic['r0'], tuple):
            raise TypeError('\n\nKey "r0" must be a tuple.\n')
        if len(dic['r0']) not in (2, 3):
            raise ValueError('\n\nKey "r0" must contain a tuple of length two (or three, '
                                      'for a lattice in 3D space).\n')
        if not all(isinstance(c, (int, float)) for c in dic['r0']):
            raise ValueError('\n\nKey "r0" must contain a tuple of real numbers.\n')
    if len({len(dic['r0']) for dic in unit_cell}) > 1:
        raise ValueError('\n\nAll "r0" must have the same length.\n')

    
def prim_vec(prim_vec):
    '''
    Check parameter *prim_vec*.

    :raises TypeError: Parameter prim_vec must be a list.
    :raises ValueError: Parameter prim_vec must be a list
      of length 1 for 1D lattices or length 2 fro 2D lattices.
    :raises TypeError: List elements must be tuples.
    :raises ValueError: List elements must be 1 or 2 tuples.
    :raises ValueError: Tuples must be of length 2.
    :raises ValueError: Tuples must containt real numbers.
    :raises ValueError: Norm of coor should be larger than 0.1.
    '''
    if not isinstance(prim_vec, list):
        raise TypeError('\n\nParameter prim_vec must be a list.\n')
    if len(prim_vec) not in (1, 2, 3):
        raise ValueError('\n\nParameter prim_vec must be a list of length 1 '
                                  'for 1D lattices, 2 for 2D lattices, or 3 for 3D lattices.\n')
    for coor in prim_vec:
        if not isinstance(coor, tuple):
            raise TypeError('\n\nParameter prim_vec must contain tuples.\n')
        if len(coor) not in (2, 3):
            raise ValueError('\n\nParameter prim_vec must contain tuples of length 2 '
                                      '(or 3, for a lattice in 3D space).\n')
        if not all(isinstance(c, (int, float)) for c in coor):
            raise ValueError('\n\nParameter prim_vec must contain tuples of '
                                      'real numbers.\n')
        if sum(c ** 2 for c in coor) < 0.1:
            raise ValueError('\n\nEach primitive vector must have a norm larger '
                                      'than 0.1.\n')
    if len({len(coor) for coor in prim_vec}) > 1:
        raise ValueError('\n\nAll primitive vectors must have the same length.\n')
    if len(prim_vec) > len(prim_vec[0]):
        raise ValueError('\n\nA lattice cannot have more primitive vectors than '
                                  'space dimensions (use 3-tuples for a 3D lattice).\n')


def independent(prim_vec):
    '''
    Check that the primitive vectors are linearly independent (needed for
    the reciprocal lattice).

    :raises ValueError: The primitive vectors must be linearly independent.
    '''
    if np.linalg.matrix_rank(np.array(prim_vec, dtype='f8'), tol=1e-8) != len(prim_vec):
        raise ValueError('\n\nThe primitive vectors must be linearly independent.\n')


def space_dim(unit_cell, prim_vec):
    '''
    Check that the unit cell positions and the primitive vectors live in the
    same space (2D or 3D).

    :raises ValueError: "r0" and prim_vec tuples must have the same length.
    '''
    if unit_cell and len(unit_cell[0]['r0']) != len(prim_vec[0]):
        raise ValueError('\n\nThe "r0" positions and the primitive vectors must have '
                                  'the same number of components.\n')


def get_lattice(prim_vec, n1, n2, n3=1):
    '''
    Check method *get_lattice*.

    :raises TypeError: Parameters n1, n2, n3 must be integers.
    :raises ValueError: Parameters n1, n2, n3 must be positive integers.
    :raises ValueError: n2 (n3) must be 1 with fewer than 2 (3) primitive vectors.
    '''
    positive_int(n1, 'n1')
    positive_int(n2, 'n2')
    positive_int(n3, 'n3')
    if len(prim_vec) == 1 and n2 > 1:
        raise ValueError('\n\nParameter n2 should be equal to 1\n'
                                    'if one primitive vector is given\n')
    if len(prim_vec) < 3 and n3 > 1:
        raise ValueError('\n\nParameter n3 should be equal to 1\n'
                                    'if fewer than three primitive vectors are given\n')


def coor(coor, dtype=None):
    '''
    Check if *coor* is a structured array with
    dtype=[('x', 'f8'), ('y', 'f8'), ('tag', 'U1')], or, for a lattice in 3D
    space, dtype=[('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('tag', 'U1')]. If
    *dtype* is given, *coor* must have exactly that dtype.
    '''
    dtypes = [np.dtype([('x', 'f8'), ('y', 'f8'), ('tag', 'U1')]),
                  np.dtype([('x', 'f8'), ('y', 'f8'), ('z', 'f8'), ('tag', 'U1')])]
    if dtype is not None:
        dtypes = [np.dtype(dtype)]
    if coor.dtype not in dtypes:
        raise TypeError('\n\nParameter coor dtype must be\n'
                                  'dtype=[("x", "f8"), ("y", "f8"), ("tag", "U1")]\n'
                                  '(with a ("z", "f8") field after "y" for a lattice in 3D space).\n')


def coor_1d(coor):
    '''
    Check if *coor* is 1d (coor['y'] = cst).
    :raises ValueError: *coor* must be 1d( coor['y'] = cst)..
    '''
    if not np.allclose(coor['y'][0], coor['y']):
        raise ValueError('\n\ncoor["y"] must be constant.\n')


def remove_sites(index, sites):
    '''
    Check method *remove_sites*.

    :raises TypeError: Parameter index must be a list.
    :raises ValueError: Parameter index must be a list of integers.
    :raises ValueError: Indices must be between 0 and sites -1.
      of integers between 0 and sites
    '''
    if not isinstance(index, list):
        raise TypeError('\n\nParameter index must be a list.\n')
    if not all(isinstance(i, int) for i in index):
        raise ValueError('\n\nParameter index must be a list of integers.\n')
    if not all(-1 < i < sites for i in index):
        raise ValueError('\n\nElements of index must be between 0 and sites - 1.\n')


def shift(shift):
    '''
    Check *shift_x* and *shift_y*.
    :raises TypeError: Parameter delta must be a real number.
    '''
    if not isinstance(shift, (int, float)):
        raise TypeError('\n\nParameter shift must be a real number.\n')


def boundary_line(cx, cy, co):
    '''
    Check *boundary_line*.
    :raises TypeError: Parameter cx must be a real number.
    :raises TypeError: Parameter cy must be a real number.
    :raises TypeError: Parameter co must be a real number.
    '''
    if not isinstance(cx, (int, float)):
        raise TypeError('\n\nParameter cx must be a real number.\n')
    if not isinstance(cy, (int, float)):
        raise TypeError('\n\nParameter cy must be a real number.\n')
    if not isinstance(co, (int, float)):
        raise TypeError('\n\nParameter co must be a real number.\n')


def space_3d(space_dim):
    '''
    Check that the lattice lives in 3D space.

    :raises ValueError: This method requires a lattice in 3D space.
    '''
    if space_dim != 3:
        raise ValueError('\n\nThis method requires a lattice in 3D space '
                                    '(unit_cell and prim_vec given as 3-tuples).\n')


def sites(sites):
    '''
    Check if *get_lattice* has been called (*coor* not empty).
    :raises RuntimeError: Run method lat.get_lattice first.
    '''
    if sites == 0:
        raise RuntimeError('\n\nRun method lat.get_lattice first.\n')


####################################
# CLASS SYSTEM EXCEPTION HANDLING
####################################


def sys(sys):
    '''
    Check if parameter is an instance of the *system*.
    :raises TypeError: Parameter must be an instance of the class system.
    '''
    names = inspect.getmro(sys.__class__)
    if str(names).find('system') == -1:
        raise TypeError('\n\nParameter must be an instance of the class system.\n')


def print_hopping(n, nmax):
    '''
    Check method *print_vec_hopping*.

    :raises TypeError: Parameter *nmax* must be an integer.
    :raises ValueError: Parameter *nmax* must be a positive integer.
      between 1 and n_max-1.
    '''
    if not isinstance(n, int):
        raise TypeError('\n\nParameter n_max must be an integer.\n')
    if n < 1 or n > nmax-1:
        raise ValueError('\n\nParameter n_max must be a positive integer'
                                    'between 1 and n_max-1.\n')


def set_onsite(onsite, tags):
    '''
    Check method *set_onsite*.

    :raises TypeError: Parameter onsite must be a dictionary.
    :raises ValueError: Parameter onsite keys must be a tag.
    :raises ValueError: Parameter onsite values must be
      real and/or complex numbers.
    '''
    if not isinstance(onsite, dict):
        raise TypeError('\n\nParameter onsite must be a dictionary.\n')
    for tag, val in onsite.items():
        if tag not in tags:
            raise ValueError('\n\nParameter onsite keys must be a tag.\n')   
        if not isinstance(val, (int, float, complex)):
            raise ValueError('\n\nParameter onsite values must be\n'\
                                       'real and/or complex numbers.\n')


def set_hopping(list_hop, n_max):
    '''
    Check method *set_hopping*.

    :raises TypeError: Parameter *list_hop* must be a list.
    :raises TypeError: Parameter *list_hop* must be a list of dictionary.
    :raises KeyError: "n" and "t" must be dictionary keys.
    :raises KeyError: "tag" or "ang" must be a key.
    :raises KeyError: "tag" and "ang" must be keys.
    :raises ValueError: Dictionaries must be of length 2, 3, or 4.
    :raises ValueError: "n" must be between 1 and nmax"

    '''
    if not isinstance(list_hop, list):
        raise TypeError('\n\nParameter *list_hop* must be a list.\n')
    for dic in list_hop:
        if not isinstance(dic, dict):
            raise TypeError('\n\nParameter *list_hop* must be a list of dictionary.\n')
        if 'n' not in dic or 't' not in dic:
                raise KeyError('\n\n"n" and "t" must be dictionary keys.\n')
        if not isinstance(dic['n'], int):
            raise TypeError('\n\n"n" value must be an integer.\n')
        if not 0 < dic['n'] <= n_max:
            raise ValueError('\n\n"n" value must be between 1 and {}.\n'.format(n_max))
        if not isinstance(dic['t'], (int, float, complex)):
            raise TypeError('\n\n"t" value must be a real or complex number.\n')
        if len(dic) == 3:
            if 'tag' not in dic and 'ang' not in dic:
                raise KeyError('\n\n"tag" or "ang" must be a key.\n')
        elif len(dic) == 4:
            if 'tag' not in dic or 'ang' not in dic:
                raise KeyError('\n\n"tag" and "ang" must be keys.\n')
        elif len(dic) > 4:
            raise ValueError('\n\nDictionaries must be of length 2, 3, or 4.\n')
        if 'tag' in dic:
            if not isinstance(dic['tag'], str):
                raise TypeError('\n\n"tag" value must be a string.\n')
            if len(dic['tag']) != 2:
                raise ValueError('\n\n"tag" value must be a string of length 2.\n')
        if 'ang' in dic:
            if not isinstance(dic['ang'], (int, float)):
                raise TypeError('\n\n"ang" value must be a real number.\n')


def index(ind, dic):
    '''
    check if *ind* not empy.
    '''
    if np.sum(ind) == 0:
        raise ValueError('\n\nNo hoppings with parameters {}.\n'.format(dic))


def set_hopping_def(hop, hopping_def, sites, var_name='hopping_def', same_site=False):
    '''
    Check methods *set_hopping_def* and *set_hopping_manual*.

    *hop* is not used (kept for backward compatibility). Indices may be
    Python or NumPy integers, values Python or NumPy numbers. If *same_site*
    is True, keys :math:`(i, i)` are accepted.

    :raises TypeError: Parameter *hopping_def* must be a dictionary
    :raises TypeError: *hopping_def* keys must be tuples.
    :raises TypeError: *hopping_def* keys must be tuples of length 2.
    :raises ValueError: *hopping_def* keys must be tuples of integers.
    :raises ValueError: *hopping_def* keys must be integers between 0 and sites-1.
    :raises ValueError: *hopping_def* keys must be different integers between 0 and sites-1.
    :raises TypeError: *hopping_def* values must be numbers.
    '''
    if not isinstance(hopping_def, dict):
        raise TypeError('\n\nParameter {} must be a dictionary.\n'.format(var_name))
    for key, val in hopping_def.items():
        if not isinstance(key, tuple):
            raise TypeError('\n\n{} keys must be tuples.\n'.format(var_name))
        if len(key) != 2:
            raise TypeError('\n\n{} keys must be tuples of length 2.\n'.format(var_name))
        if not isinstance(key[0], (int, np.integer)) or not isinstance(key[1], (int, np.integer)):
            raise ValueError('\n\n{} keys must be tuples of integers.\n'.format(var_name))
        if key[0] < 0 or key[1] < 0 or key[0] > sites-1 or key[1] > sites-1:
            raise ValueError('\n\n{} keys must be integers between 0 and sites-1.\n'.format(var_name))
        if key[0] == key[1] and not same_site:
            raise ValueError('\n\n{} keys must be different integers between 0 and sites-1.\n'.format(var_name))
        if not isinstance(val, (int, float, complex, np.number)):
            raise TypeError('\n\n{} values must be numbers.\n'.format(var_name))


def set_onsite_def(onsite_def, sites):
    '''
    Check method *set_ons_def*.

    :raises TypeError: Parameter *onsite_def* must be a dictionary.
    :raises TypeError: *onsite_def* keys must be integers.
    :raises TypeError: *onsite_def* values must be numbers.
    :raises ValueError: *onsite_def* keys must be integers between :math:`[0, sites)`.
    '''
    if not isinstance(onsite_def, dict):
        raise TypeError('\n\nParameter onsite_def must be a dictionary.\n')
    for key, val in onsite_def.items():
        if not isinstance(key, int):
            raise TypeError('\n\nonsite_def keys must be integers.\n')
        if not isinstance(val, (int, float, complex)):
            raise TypeError('\n\nonsite_def values must be numbers.\n')
        if key < 0 or key > sites-1:
            raise ValueError('\n\nonsite_def keys must be integers between 0 and sites-1.\n')


def hop_n1(hop):
    '''
    Check method if self.hop contains nearest neighbours hoppings.

    :raises ValueError: self.hop must contain nearest neighbours hoppings.
    '''
    if not np.any(hop['n'] == 1):
        raise ValueError('\n\nParameter hop must contain nearest neighbours hoppings.\n')


def empty_onsite(onsite):
    '''
    Check if *onsite* not empty.

    :raises RuntimeError: Run method set_onsite first.
    '''
    if onsite.size == 0:
        raise RuntimeError('\n\nRun method set_onsite first\n')


def empty_hop(hop):
    '''
    Check if *hop* not empty.

    :raises RuntimeError: Run method set_hopping first.
    '''
    if hop.size == 0:
        raise RuntimeError('\n\nRun method set_hopping first\n')


def hop_sites(hop, sites):
    '''
    Check if *hop* indices are smaller than *sites*.

    :raises ValueError: Run method sys.clear_hopping and redefine the hoppings.
    '''
    ind_max = max(np.max(hop[f]) for f in ('i', 'j') if f in hop.dtype.names)
    if ind_max >= sites:
        raise ValueError('\n\nThe hoppings refer to sites that no longer exist.\n'
                                    'Run method system.clear_hopping '
                                    'and redefine the hoppings.\n')


def empty_coor(coor):
    '''
    Check if *coor* not empty.

    :raises RuntimeError: Run method lattice.get_lattice first.
    '''
    if coor.size == 0:
        raise RuntimeError('\n\nRun method lattice.get_lattice first.\n')


def empty_coor_hop(coor_hop):
    '''
    Check if *coor_hop* not empty.

    :raises RuntimeError: Run method system.get_coor_hop first.
    '''
    if coor_hop.size == 0:
        raise RuntimeError('\n\nRun method system.get_coor_hop first.\n')


def empty_ham(ham):
    '''
    Check if Hamiltonian not empty.

    :raises RuntimeError: Run method system.get_ham first.
    '''
    if not ham.nnz:
        raise RuntimeError('\n\nRun method system.get_ham first.\n')


def empty_en(en):
    '''
    Check if *en* not empty.

    :raises RuntimeError: Run method get_ham first.
    '''
    if en.size == 0:
        raise RuntimeError('\n\nRun method get_eig first\n')


def empty_pola(pola):
    '''
    Check if *pola* not empty.

    :raises RuntimeError: Run method get_eig(eigenvec=True) first.
    '''
    if pola.size == 0:
        raise RuntimeError('\n\nRun method get_eig(eigenvec=True) first\n')


def empty_vn(vn):
    '''
    Check if *vn* not empty.

    :raises RuntimeError: Run method get_eig(eigenvec=True) first.
    '''
    if vn.size == 0:
        raise RuntimeError('\n\nRun method get_eig(eigenvec=True) first\n')


def empty_ipr(ipr):
    '''
    Check if *ipr* not empty.
    '''
    if ipr.size == 0:
        raise RuntimeError('\n\nRun method get_ipr first\n')


def empty_ndarray(arr, method):
    '''
    Check if *arr* is a not empty np.ndarray.
    '''
    if arr.size == 0:
        raise RuntimeError('\n\nRun method {} first\n'. format(method))


def tag(tag, tags):
    '''
    Check tag.

    :raises TypeError: Parameter *tag* must be a string.
    :raises ValueError: Parameter *tag* is not in tags.
    '''
    if not isinstance(tag, str):
        raise TypeError('\n\nParameter tag must be a one-character string.\n')
    if tag not in tags:
        raise ValueError('\n\nParameter tag is not in tags.\n')


def angle(angle, angles, upper_part):
    '''
    Check angle.

    :raises TypeError: Parameter *angle* must be

        * a positive number if *upper_part* is True
        * a negative real if *upper_part* is False.
    :raises ValueError: Parameter *angle* is not in hop['ang'].
    '''
    if upper_part:
        positive_real_zero(angle, 'angle, if upper_part=True,')
    else:
        negative_real(angle, 'angle, if upper_part=False,')
    if not np.sum(np.isclose(angle, angles, atol=ATOL)) and \
            not np.sum(np.isclose(angle, angles - 180, atol=ATOL)):
        raise ValueError('\n\nParameter angle is not in hop["ang"].\n')
    

def lims(lims):
    '''
    Check parameter *lims*.

    :raises TypeError: Parameter lims must be a list.
    :raises TypeError: Parameter *lims[0]* must be a real number.
    :raises TypeError: Parameter *lims[1]* must be a real number.
    :raises ValueError: *lims* must be a list of length 2.
    :raises ValueError: *lims[0]* must be smaller than *lims[1]*.
    '''
    if lims is not None:
        list_tuple_2elem(lims, 'lims')
        real_number(lims[0], 'lims[0]')
        real_number(lims[1], 'lims[1]')
        smaller(lims[0], 'lims[0]', lims[1], 'lims[1]')


def lims_positive(lims):
    '''
    Check parameter *lims*.

    :raises TypeError: Parameter lims must be a list.
    :raises TypeError: Parameter *lims[0]* must be a positive real number.
    :raises TypeError: Parameter *lims[1]* must be a positive real number.
    :raises ValueError: *lims* must be a list of length 2.
    :raises ValueError: *lims[0]* must be smaller than *lims[1]*.
    '''
    if lims is not None:
        list_tuple_2elem(lims, 'lims')
        positive_real(lims[0], 'lims[0]')
        positive_real(lims[1], 'lims[1]')
        smaller(lims[0], 'lims[0]', lims[1], 'lims[1]')


#################################
# CLASS PLOT EXCEPTION HANDLING
#################################


def fig(fig):
    '''
    Check if fig is an instance of *Figure*.

    :raises TypeError: fig must be an instance of *Figure*.
    '''
    if not fig.__class__.__name__ == 'Figure':
        raise TypeError('\n\nfig must be an instance of *Figure*.\n')


def ani(ani):
    '''
    Check if ani is an instance of *FuncAnimation*.

    :raises TypeError: ani must be an instance of *FuncAnimation*.
    '''
    if not ani.__class__.__name__ == 'FuncAnimation':
        raise TypeError('\n\nani must be an instance of *FuncAnimation*.\n')


def file_format(file_format):
    '''
    Check if file_format is a string 'png', 'pdf', 'ps', 'eps', or 'svg'.

    :raises TypeError: file_format must be a string.
    :raises ValueError: file_format must be a string given by 'png', 'pdf', 'ps', 'eps', or 'svg'.

    '''
    if not isinstance(file_format, str):
        raise TypeError('\n\nfile_format must be a string.\n')
    if file_format not in ['png', 'pdf', 'ps', 'eps', 'svg']:
        raise ValueError('\n\nfile_format must be a string given by,\n'\
                                   ' "png", "pdf", "ps", "eps", or "svg".\n')


####################################
# PROPAGATION
####################################


def get_pump(hams):
    '''
    Check method *propagation.get_pumping*.

    :raises TypeError: hams must be a non-empty list.
    :raises RuntimeError: Run method system.get_ham first.
    '''
    if not isinstance(hams, list) or not hams:
        raise TypeError('\n\nhams must be a non-empty list.\n')
    for ham in hams:
        empty_ham(ham)

def prop_type(prop_type):
    string(prop_type, 'prop_type')
    if prop_type not in ['real', 'imag', 'norm']:
        raise ValueError('\n\nParameter prop_type must be a string:\n'
                                   '"real", "imag", "norm".\n')


####################################
# CLASS KSPACE EXCEPTION HANDLING
####################################


def k_vector(vec, var_name, ndim):
    '''
    Check that *vec* is a tuple/list of *ndim* real numbers.

    :raises TypeError: Parameter *var_name* must be a tuple/list.
    :raises ValueError: Parameter *var_name* must be of length *ndim*.
    :raises TypeError: Parameter *var_name* must contain real numbers.
    '''
    if not isinstance(vec, (tuple, list, np.ndarray)):
        raise TypeError('\n\nParameter {} must be a tuple, list, or ndarray.\n'.format(var_name))
    if len(vec) != ndim:
        raise ValueError('\n\nParameter {} must be of length {}.\n'.format(var_name, ndim))
    for val in vec:
        if not isinstance(val, (int, float, np.integer, np.floating)):
            raise TypeError('\n\nParameter {} must contain real numbers.\n'.format(var_name))


def ks(ks, ndim):
    '''
    Check the k-points passed to *kspace.get_bands*: a 2D array of shape
    (nk, *ndim*).

    :raises ValueError: Parameter ks must be of shape (nk, ndim).
    '''
    if ks.ndim != 2 or ks.shape[1] != ndim:
        raise ValueError('\n\nParameter ks must be an array of shape (nk, {}).\n'.format(ndim))


def spin_matrix(t, var_name):
    '''
    Check a spinful "t" or onsite value: either a plain number, or a 2x2
    complex matrix.

    :raises TypeError: Parameter *var_name* must be a number or a 2x2 matrix.
    '''
    if isinstance(t, (int, float, complex)):
        return
    t = np.asarray(t)
    if t.shape != (2, 2):
        raise TypeError('\n\nParameter {} must be a number or a 2x2 matrix.\n'.format(var_name))


def set_hopping_kspace(list_hop, n_sites, ndim, spin=False):
    '''
    Check method *kspace.set_hopping*.

    :raises TypeError: Parameter *list_hop* must be a list of dictionaries.
    :raises KeyError: "i", "j", "R", and "t" must be dictionary keys.
    :raises ValueError: "i" and "j" must be site indices between 0 and n_sites-1.
    :raises ValueError: "R" must be a tuple of *ndim* integers.
    :raises TypeError: "t" must be a real or complex number (or, if *spin*,
      a 2x2 matrix).
    '''
    if not isinstance(list_hop, list):
        raise TypeError('\n\nParameter list_hop must be a list.\n')
    for dic in list_hop:
        if not isinstance(dic, dict):
            raise TypeError('\n\nParameter list_hop must be a list of dictionaries.\n')
        if not {'i', 'j', 'R', 't'} <= set(dic):
            raise KeyError('\n\n"i", "j", "R", and "t" must be dictionary keys.\n')
        if not isinstance(dic['i'], int) or not isinstance(dic['j'], int):
            raise TypeError('\n\n"i" and "j" must be integers.\n')
        if not (0 <= dic['i'] < n_sites) or not (0 <= dic['j'] < n_sites):
            raise ValueError('\n\n"i" and "j" must be site indices between 0 and {}.\n'.format(n_sites-1))
        if not isinstance(dic['R'], tuple) or len(dic['R']) != ndim:
            raise ValueError('\n\n"R" must be a tuple of {} integers.\n'.format(ndim))
        if not all(isinstance(n, int) for n in dic['R']):
            raise TypeError('\n\n"R" must be a tuple of integers.\n')
        if dic['i'] == dic['j'] and dic['R'] == (0,) * ndim:
            raise ValueError('\n\nUse kspace.set_onsite for i == j and R == 0 '
                                  '(it accepts a 2x2 spin matrix when spin=True).\n')
        if spin:
            spin_matrix(dic['t'], '"t"')
        elif not isinstance(dic['t'], (int, float, complex)):
            raise TypeError('\n\n"t" value must be a real or complex number.\n')


def set_onsite_kspace(dict_onsite, tags, spin=False):
    '''
    Check method *kspace.set_onsite*.

    :raises TypeError: Parameter *dict_onsite* must be a dictionary.
    :raises ValueError: keys must be tags.
    :raises TypeError: values must be real or complex numbers (or, if
      *spin*, a pair of real/complex numbers, or a 2x2 matrix).
    '''
    if not isinstance(dict_onsite, dict):
        raise TypeError('\n\nParameter dict_onsite must be a dictionary.\n')
    for tag, val in dict_onsite.items():
        if tag not in tags:
            raise ValueError('\n\nParameter dict_onsite keys must be a tag.\n')
        if spin and not isinstance(val, (int, float, complex)):
            is_pair = (isinstance(val, (tuple, list)) and len(val) == 2
                            and all(isinstance(v, (int, float, complex)) for v in val))
            is_mat = np.ndim(val) == 2 and np.shape(val) == (2, 2)
            if not (is_pair or is_mat):
                raise TypeError('\n\nParameter dict_onsite values must be a number, or, '
                                           'if spin, a pair of numbers (E_up, E_down) or a '
                                           '2x2 matrix.\n')
        elif not spin and not isinstance(val, (int, float, complex)):
            raise TypeError('\n\nParameter dict_onsite values must be real and/or complex numbers.\n')


def k_path_points(points, ndim):
    '''
    Check parameter *points* used by *kspace.k_path*.

    :raises TypeError: Parameter points must be a list.
    :raises ValueError: Parameter points must contain at least two k-points.
    '''
    if not isinstance(points, list):
        raise TypeError('\n\nParameter points must be a list of k-points.\n')
    if len(points) < 2:
        raise ValueError('\n\nParameter points must contain at least two k-points.\n')
    for i, pt in enumerate(points):
        k_vector(pt, 'points[{}]'.format(i), ndim)


####################################
# DENSITY OF STATES
####################################


def dos_kernel(kernel):
    '''
    Check parameter *kernel* used by *dos.density_of_states*.

    :raises TypeError: Parameter kernel must be a string.
    :raises ValueError: Parameter kernel must be "gaussian" or "lorentzian".
    '''
    string(kernel, 'kernel')
    if kernel not in ['gaussian', 'lorentzian']:
        raise ValueError('\n\nParameter kernel must be a string:\n'
                                   '"gaussian", "lorentzian".\n')


def nk(nk, ndim):
    '''
    Check parameter *nk* used by *kspace.mesh_bands* / *kspace.berry_curvature*.

    :raises TypeError: Parameter nk must be an integer or a tuple of integers.
    :raises ValueError: Parameter nk (or each of its elements) must be a
        positive integer.
    :raises ValueError: Parameter nk must be a tuple of length *ndim*.
    '''
    if isinstance(nk, int):
        positive_int(nk, 'nk')
        return
    if not isinstance(nk, tuple):
        raise TypeError('\n\nParameter nk must be an integer or a tuple of integers.\n')
    if len(nk) != ndim:
        raise ValueError('\n\nParameter nk must be of length {}.\n'.format(ndim))
    for n in nk:
        positive_int(n, 'nk')


####################################
# TOPOLOGY
####################################


def dim_2(dim):
    '''
    Check that the model is 2D. Berry curvature / Chern number are only
    defined for a 2D Brillouin zone.

    :raises ValueError: This calculation requires a 2D lattice.
    '''
    if dim != 2:
        raise ValueError('\n\nThis calculation requires a 2D lattice '
                                    '(two primitive vectors).\n')


def dim_min(dim, dmin):
    '''
    Check that the model has at least *dmin* periodic dimensions.

    :raises ValueError: This calculation requires a lattice with at least dmin primitive vectors.
    '''
    if dim < dmin:
        raise ValueError('\n\nThis calculation requires a lattice with at least {} '
                                    'primitive vectors.\n'.format(dmin))


def plane(plane, dim):
    '''
    Check parameter *plane* of *kspace.berry_curvature*: two distinct
    reciprocal-vector indices below *dim*.

    :raises TypeError: Parameter plane must be a tuple of two integers.
    :raises ValueError: Parameter plane must contain two distinct indices below dim.
    '''
    if not isinstance(plane, tuple) or len(plane) != 2 or \
            not all(isinstance(p, int) for p in plane):
        raise TypeError('\n\nParameter plane must be a tuple of two integers.\n')
    if plane[0] == plane[1] or not all(0 <= p < dim for p in plane):
        raise ValueError('\n\nParameter plane must contain two distinct indices '
                                    'between 0 and {}.\n'.format(dim - 1))


def direction(direction, dim=2):
    '''
    Check parameter *direction* used by *kspace.ribbon* (and the Wilson-loop
    methods): a primitive-vector index below *dim*.

    :raises TypeError: Parameter direction must be an integer.
    :raises ValueError: Parameter direction must be between 0 and dim-1.
    '''
    if not isinstance(direction, int):
        raise TypeError('\n\nParameter direction must be an integer.\n')
    if not 0 <= direction < dim:
        raise ValueError('\n\nParameter direction must be between 0 and {}.\n'.format(dim - 1))


def band_indices(bands, norb):
    '''
    Check parameter *bands* used by *kspace.berry_curvature*.

    :raises TypeError: Parameter bands must be a non-empty list of integers.
    :raises ValueError: Parameter bands must be a list of distinct band
        indices between 0 and norb-1.
    '''
    if not isinstance(bands, list) or not bands:
        raise TypeError('\n\nParameter bands must be a non-empty list of integers.\n')
    if not all(isinstance(b, int) for b in bands):
        raise TypeError('\n\nParameter bands must be a non-empty list of integers.\n')
    if len(set(bands)) != len(bands):
        raise ValueError('\n\nParameter bands must be a list of distinct band indices.\n')
    if not all(0 <= b < norb for b in bands):
        raise ValueError('\n\nParameter bands must be integers between 0 and {}.\n'.format(norb-1))


####################################
# BERRY PHASES, Z2, SYMMETRIES
####################################


def k_perp(k_perp, n):
    '''
    Check parameter *k_perp* of the Wilson-loop methods: a tuple of *n* real
    numbers (fractional coordinates along the other reciprocal vectors).

    :raises ValueError: Parameter k_perp must contain n real numbers.
    '''
    if len(k_perp) != n or not all(isinstance(f, (int, float)) for f in k_perp):
        raise ValueError('\n\nParameter k_perp must contain {} real number(s).\n'.format(n))


def flow(flow, direction, dim):
    '''
    Check parameter *flow* of *kspace.wannier_flow*.

    :raises TypeError: Parameter flow must be an integer.
    :raises ValueError: Parameter flow must be a direction other than *direction*.
    '''
    if not isinstance(flow, int):
        raise TypeError('\n\nParameter flow must be an integer.\n')
    if flow == direction or not 0 <= flow < dim:
        raise ValueError('\n\nParameter flow must be a direction between 0 and {}, '
                                    'other than direction.\n'.format(dim - 1))


def even_bands(bands):
    '''
    Check that *bands* is a list of an even number of band indices (Kramers pairs).

    :raises TypeError: Parameter bands must be a list.
    :raises ValueError: Parameter bands must contain an even number of bands.
    '''
    if not isinstance(bands, list):
        raise TypeError('\n\nParameter bands must be a list of band indices.\n')
    if len(bands) % 2:
        raise ValueError('\n\nParameter bands must contain an even number of bands '
                                    '(Kramers pairs).\n')


def operator(mat, norb):
    '''
    Check a symmetry operator: a (norb, norb) matrix.

    :raises ValueError: The operator must be a (norb, norb) matrix.
    '''
    if mat.shape != (norb, norb):
        raise ValueError('\n\nThe operator must be a ({0}, {0}) matrix.\n'.format(norb))


def parities(xi, tol=1e-6):
    '''
    Check that inversion eigenvalues are +-1.

    :raises ValueError: The bands are not an inversion-invariant subspace.
    '''
    if not np.allclose(np.abs(xi.real), 1., atol=tol) or not np.allclose(xi.imag, 0., atol=tol):
        raise ValueError('\n\nThe parity eigenvalues are not +-1: the operator is not '
                                    'an inversion symmetry of these bands (or the bands are '
                                    'degenerate with the others at a TRIM).\n')
    if np.sum(xi.real < 0) % 2:
        raise ValueError('\n\nOdd number of negative parities: the bands are not '
                                    'Kramers pairs.\n')


def k_map(k_map):
    '''
    Check parameter *k_map* of *kspace.symmetry_error*.

    :raises ValueError: Parameter k_map must be 'minus' or 'identity'.
    '''
    if k_map not in ('minus', 'identity'):
        raise ValueError('\n\nParameter k_map must be "minus" or "identity".\n')


def is_symmetry(err, tol):
    '''
    Check that a symmetry error is below *tol*.

    :raises ValueError: The operator is not a symmetry of the model.
    '''
    if err > tol:
        raise ValueError('\n\nThe operator is not a symmetry of the model '
                                    '(deviation {:.3g}).\n'.format(err))


def square(sq, tol):
    '''
    Check that the square of an antiunitary operator is +-1.

    :raises ValueError: The square of the operator must be +1 or -1.
    '''
    if not np.allclose(sq, sq[0, 0] * np.eye(len(sq)), atol=tol) or \
            not np.isclose(abs(sq[0, 0]), 1., atol=tol):
        raise ValueError('\n\nThe square of the antiunitary operator must be +1 or -1.\n')


def hermitian(ham):
    '''
    Check that a (sparse) Hamiltonian is Hermitian.

    :raises ValueError: This calculation requires a Hermitian Hamiltonian.
    '''
    if (abs(ham - ham.conj().T) > 1e-12).nnz:
        raise ValueError('\n\nThis calculation requires a Hermitian Hamiltonian.\n')


def planar_cell(prim_vec):
    '''
    Check that the lattice has two in-plane primitive vectors (a unit-cell area).

    :raises ValueError: Give the area per site explicitly.
    '''
    if len(prim_vec) != 2 or len(prim_vec[0]) != 2:
        raise ValueError('\n\nThe unit-cell area needs two in-plane primitive vectors: '
                                    'give the area per site explicitly.\n')


def integer(var, var_name):
    '''
    Check if *var* is an integer.

    :raises TypeError: Parameter *var* must be an integer.
    '''
    if not isinstance(var, int):
        raise TypeError('\n\nParameter {} must be an integer.\n'.format(var_name))


def dim_exact(dim, d):
    '''
    Check that the model has exactly *d* periodic dimensions.

    :raises ValueError: This calculation requires a lattice with d primitive vectors.
    '''
    if dim != d:
        raise ValueError('\n\nThis calculation requires a lattice with {} primitive '
                                    'vector(s).\n'.format(d))


def nonzero_number(var, var_name):
    '''
    Check if *var* is a nonzero number.

    :raises TypeError: Parameter *var* must be a number.
    :raises ValueError: Parameter *var* must be nonzero.
    '''
    number(var, var_name)
    if var == 0:
        raise ValueError('\n\nParameter {} must be nonzero.\n'.format(var_name))


def gbz_roots(n_roots, p):
    '''
    Check that det[H(beta) - E] has roots on both sides of the GBZ.

    :raises ValueError: The model has no hoppings in both directions.
    '''
    if p < 1 or n_roots <= p:
        raise ValueError('\n\nThe generalized Brillouin zone needs hoppings in both '
                                    'directions along the chain.\n')


####################################
# EXCEPTIONAL POINTS
####################################


def ham_model(model, cls):
    '''
    Check the model of *tbkit.exceptional*: a 2D **KSpace** instance, or a
    callable of a point of a 2D parameter plane.

    :raises TypeError: Parameter model must be a KSpace instance or a callable.
    :raises ValueError: A KSpace model must have two primitive vectors.
    '''
    if isinstance(model, cls):
        if model.dim != 2:
            raise ValueError('\n\nParameter model must be a KSpace with two primitive '
                                        'vectors (a 2D Brillouin zone).\n')
    elif not callable(model):
        raise TypeError('\n\nParameter model must be a KSpace instance, or a callable '
                                  'returning a square matrix.\n')


def ham_matrix(ham):
    '''
    Check the matrix returned by the callable model of *tbkit.exceptional*:
    square and finite.

    :raises ValueError: The model must return a finite square matrix.
    '''
    if ham.ndim != 2 or ham.shape[0] != ham.shape[1] or not np.all(np.isfinite(ham)):
        raise ValueError('\n\nThe model must return a finite square matrix.\n')


def loop(loop):
    '''
    Check a closed loop of *tbkit.exceptional*: a callable of s in [0, 1]
    returning a point of the plane, or the vertices of a polygon.

    :raises TypeError: Parameter loop must be a callable or an array of vertices.
    :raises ValueError: A polygon needs at least 3 vertices with 2 real coordinates.
    '''
    if callable(loop):
        return
    try:
        arr = np.asarray(loop, dtype='f8')
    except (TypeError, ValueError):
        arr = np.zeros(())
    if arr.ndim == 0:
        raise TypeError('\n\nParameter loop must be a callable of s in [0, 1], or an '
                                  'array of polygon vertices.\n')
    if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] != 2 or not np.all(np.isfinite(arr)):
        raise ValueError('\n\nParameter loop must hold at least 3 vertices of 2 real '
                                    'coordinates, shape (m, 2).\n')


def loop_point(point):
    '''
    Check a point returned by a loop callable: 2 finite real numbers.

    :raises ValueError: The loop must return points of 2 real coordinates.
    '''
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        raise ValueError('\n\nParameter loop must return points of 2 finite real '
                                    'coordinates.\n')


def closed_loop(ham_start, ham_end):
    '''
    Check that a loop is closed: the Hamiltonian at its end is the one at its start.

    :raises ValueError: The loop is not closed.
    '''
    scale = max(1., float(np.max(np.abs(ham_start))))
    if np.max(np.abs(ham_end - ham_start)) > 1e-9 * scale:
        raise ValueError('\n\nParameter loop must be closed: H(loop(1)) differs from '
                                    'H(loop(0)).\n')


def band_pair(bands, norb):
    '''
    Check a pair of band labels: two distinct integers between 0 and norb-1.

    :raises TypeError: Parameter bands must be a pair of integers.
    :raises ValueError: Parameter bands must be two distinct band indices.
    '''
    if not isinstance(bands, (tuple, list)) or len(bands) != 2 or \
            not all(isinstance(b, int) for b in bands):
        raise TypeError('\n\nParameter bands must be a pair of integers (m, n).\n')
    if bands[0] == bands[1] or not all(0 <= b < norb for b in bands):
        raise ValueError('\n\nParameter bands must be two distinct band indices between 0 '
                                    'and {}.\n'.format(norb - 1))


def band_index(band, norb):
    '''
    Check a band label: an integer between 0 and norb-1.

    :raises TypeError: Parameter band must be an integer.
    :raises ValueError: Parameter band must be between 0 and norb-1.
    '''
    integer(band, 'band')
    if not 0 <= band < norb:
        raise ValueError('\n\nParameter band must be between 0 and {}.\n'.format(norb - 1))


def tracking(k):
    '''
    Eigenvalue continuation failed: the path meets a degeneracy.

    :raises ValueError: Cannot follow the eigenvalues continuously near k.
    '''
    raise ValueError('\n\nCannot follow the eigenvalues continuously near k = {}: the path '
                                'passes through (or too close to) a degeneracy. Move or '
                                'enlarge the loop.\n'.format(np.round(k, 8)))


def winding_path(k):
    '''
    The discriminant vanishes on the path of a winding number.

    :raises ValueError: The loop passes through a zero of the discriminant.
    '''
    raise ValueError('\n\nThe discriminant vanishes (to numerical precision) on the path, '
                                'near k = {}: the loop passes through a degeneracy. Move the '
                                'loop, or change nk.\n'.format(np.round(k, 8)))


def discriminant_nonzero(scale):
    '''
    Check that the discriminant does not vanish identically.

    :raises ValueError: The bands are degenerate everywhere.
    '''
    if scale == 0.:
        raise ValueError('\n\nThe discriminant vanishes everywhere: some bands are '
                                    'degenerate over the whole zone.\n')


def gap_kind(gap):
    '''
    Check the kind of line gap: 'real' (Re E) or 'imaginary' (Im E).

    :raises ValueError: Parameter gap must be "real" or "imaginary".
    '''
    if gap not in ('real', 'imaginary'):
        raise ValueError('\n\nParameter gap must be "real" or "imaginary".\n')


def biorthogonal_kind(kind):
    '''
    Check the kind of Berry connection: 'LR', 'RL', 'RR' or 'LL'.

    :raises ValueError: Parameter kind must be "LR", "RL", "RR" or "LL".
    '''
    if kind not in ('LR', 'RL', 'RR', 'LL'):
        raise ValueError('\n\nParameter kind must be "LR", "RL", "RR" or "LL".\n')


def line_gap(keys, bands, gap):
    '''
    Check that *bands* are separated from the other bands by a line gap:
    with the bands ordered by *keys* (Re E or Im E, shape (nk, norb)) at
    every k-point, a band inside the group and its neighbour outside it
    never overlap over the whole mesh.

    :raises ValueError: The bands have no line gap.
    '''
    inside = np.zeros(keys.shape[1], bool)
    inside[bands] = True
    for n in range(keys.shape[1] - 1):
        if inside[n] != inside[n + 1] and np.max(keys[:, n]) >= np.min(keys[:, n + 1]):
            part = 'Re E' if gap == 'real' else 'Im E'
            raise ValueError('\n\nThe bands {} are not separated from the others by a {} line '
                                        'gap: sorted by {}, band {} reaches band {} over the '
                                        'Brillouin zone, so the band labels are not continuous. '
                                        'For a non-Hermitian model, use '
                                        'KSpace.biorthogonal_chern_number with the other kind of '
                                        'gap, or locate the gap-closing exceptional points with '
                                        'tbkit.exceptional.find_exceptional_points.\n'
                                        .format(bands, gap, part, n, n + 1))


def band_continuity(en, bands):
    '''
    Check that a group of bands, labelled at every point of a periodic k
    mesh (*en*, complex, shape (n1, n2, norb)), is continuous: matching the
    eigenvalues of neighbouring points by distance maps the group onto
    itself. A line gap measured on the mesh alone can miss a narrow
    crossing between mesh points; this catches the jump of the labels.

    :raises ValueError: The band labels jump between neighbouring k-points.
    '''
    from scipy.optimize import linear_sum_assignment
    inside = np.zeros(en.shape[2], bool)
    inside[bands] = True
    for axis in (0, 1):
        nxt = np.roll(en, -1, axis=axis)
        for e_a, e_b in zip(en.reshape(-1, en.shape[2]), nxt.reshape(-1, en.shape[2])):
            _, perm = linear_sum_assignment(np.abs(e_a[:, None] - e_b[None, :]))
            if np.any(inside != inside[perm]):
                raise ValueError('\n\nThe bands {} are not separated from the others over the '
                                            'Brillouin zone: between neighbouring k-points they '
                                            'exchange eigenvalues with the other bands (the line gap '
                                            'closes, e.g. at exceptional points, or nk is too small). '
                                            'For a non-Hermitian model, try '
                                            'KSpace.biorthogonal_chern_number with the other kind of '
                                            'gap, or tbkit.exceptional.find_exceptional_points.\n'
                                            .format(bands))


def no_overlap(overlap):
    '''
    Check that no overlap matrix is set (non-orthogonal bases are not
    supported by the biorthogonal tools).

    :raises ValueError: Not available with an overlap matrix.
    '''
    if overlap:
        raise ValueError('\n\nThis method does not support an overlap matrix (set_overlap).\n')


def weights(weights, n):
    '''
    Check the weights of *occupation.fermi_level*: *n* positive numbers.

    :raises ValueError: Parameter weights must contain n positive numbers.
    '''
    if len(weights) != n or np.any(weights <= 0):
        raise ValueError('\n\nParameter weights must contain {} positive numbers.\n'.format(n))


def electrons(n_electrons, capacity):
    '''
    Check a number of electrons: between 0 and the total weight of the levels.

    :raises TypeError: Parameter n_electrons must be a real number.
    :raises ValueError: Parameter n_electrons must be between 0 and capacity.
    '''
    positive_real_zero(n_electrons, 'n_electrons')
    if n_electrons > capacity * (1 + 1e-12):
        raise ValueError('\n\nParameter n_electrons must be between 0 and {:g} '
                                    '(the number of states).\n'.format(capacity))


def n_eig(n_eig, sites):
    '''
    Check the number of eigenpairs of *system.get_eig_sparse*.

    :raises TypeError: Parameter n_eig must be an integer.
    :raises ValueError: Parameter n_eig must be between 1 and sites-2.
    '''
    positive_int(n_eig, 'n_eig')
    if n_eig > sites - 2:
        raise ValueError('\n\nParameter n_eig must be between 1 and {} (use get_eig for '
                                    'the full spectrum).\n'.format(sites - 2))


####################################
# TRANSPORT
####################################


def square_matrix(mat, var_name):
    '''
    Check that *mat* is a square 2D array.

    :raises ValueError: Parameter must be a square matrix.
    '''
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError('\n\nParameter {} must be a square matrix.\n'.format(var_name))


def lead(h0, v):
    '''
    Check a lead: square cell Hamiltonian *h0* and coupling *v* of the same shape.

    :raises ValueError: h0 and v must be square matrices of the same shape.
    '''
    square_matrix(h0, 'h0')
    if v.shape != h0.shape:
        raise ValueError('\n\nParameters h0 and v must be square matrices of the same shape.\n')


def converged(flag, name):
    '''
    Check that an iteration converged.

    :raises RuntimeError: The iteration did not converge.
    '''
    if not flag:
        raise RuntimeError('\n\n{} did not converge: increase max_iter.\n'.format(name))


def lead_direction(direction):
    '''
    Check the direction of a lead: +1 or -1.

    :raises ValueError: Parameter direction must be +1 or -1.
    '''
    if direction not in (1, -1):
        raise ValueError('\n\nParameter direction must be +1 or -1.\n')


def nearest_cells(n):
    '''
    Check that a lead hopping reaches at most the neighbouring cells.

    :raises ValueError: Lead hoppings must connect neighbouring cells only.
    '''
    if abs(n) > 1:
        raise ValueError('\n\nA lead from a KSpace model needs hoppings between '
                                    'neighbouring cells only (use a larger unit cell).\n')


def lead_coupling(coupling, sites, m, n_device):
    '''
    Check the coupling of a lead to the device.

    :raises TypeError: Parameter sites must be a list of integers.
    :raises ValueError: sites must be distinct device indices, coupling of shape (len(sites), m).
    '''
    if not isinstance(sites, list) or not all(isinstance(s, (int, np.integer)) for s in sites):
        raise TypeError('\n\nParameter sites must be a list of integers.\n')
    if len(set(sites)) != len(sites) or not all(0 <= s < n_device for s in sites):
        raise ValueError('\n\nParameter sites must contain distinct device indices '
                                    'between 0 and {}.\n'.format(n_device - 1))
    if coupling.shape != (len(sites), m):
        raise ValueError('\n\nParameter coupling must be of shape (len(sites), len(h0)) '
                                    '= ({}, {}).\n'.format(len(sites), m))


def lead_index(lead, n_leads):
    '''
    Check a lead index.

    :raises TypeError: The lead index must be an integer.
    :raises ValueError: No such lead.
    '''
    if not isinstance(lead, int):
        raise TypeError('\n\nThe lead index must be an integer.\n')
    if not 0 <= lead < n_leads:
        raise ValueError('\n\nNo lead {} ({} lead(s) attached with add_lead).\n'.format(lead, n_leads))



####################################
# KERNEL POLYNOMIAL METHOD
####################################


def kpm_kernel(kernel):
    '''
    Check the KPM kernel name.

    :raises ValueError: Parameter kernel must be "jackson" or "lorentz".
    '''
    if kernel not in ('jackson', 'lorentz'):
        raise ValueError('\n\nParameter kernel must be "jackson" or "lorentz".\n')


def kpm_grid(eps):
    '''
    Check that the KPM energies lie inside the (scaled) spectral bounds.

    :raises ValueError: e_grid must lie inside the spectral bounds.
    '''
    if np.any(np.abs(eps) >= 1.):
        raise ValueError('\n\nParameter e_grid must lie strictly inside the spectral '
                                    'bounds.\n')


def site_index(site, n):
    '''
    Check a site (row) index.

    :raises TypeError: Parameter site must be an integer.
    :raises ValueError: Parameter site must be between 0 and n-1.
    '''
    if not isinstance(site, (int, np.integer)):
        raise TypeError('\n\nParameter site must be an integer.\n')
    if not 0 <= site < n:
        raise ValueError('\n\nParameter site must be between 0 and {}.\n'.format(n - 1))


def max_order(list_hop):
    '''
    Largest neighbour order 'n' requested by a (possibly malformed) list of
    hoppings; 1 if there is none (the list is validated afterwards).
    '''
    try:
        orders = [dic['n'] for dic in list_hop if isinstance(dic['n'], int)]
    except (TypeError, KeyError):
        return 1
    return max([n for n in orders if n > 0] + [1])


####################################
# ORBITALS, SLATER-KOSTER, SPIN
####################################


def orbitals(orbs):
    '''
    Check a list of orbital names.

    :raises TypeError: The orbitals must be a list of names.
    :raises ValueError: Unknown or repeated orbital.
    '''
    from tbkit.slater_koster import ORBITALS
    if not isinstance(orbs, list) or not orbs:
        raise TypeError('\n\nThe orbitals must be a non-empty list of names.\n')
    for o in orbs:
        if o not in ORBITALS:
            raise ValueError('\n\nUnknown orbital {!r}: use one of {}.\n'.format(o, list(ORBITALS)))
    if len(set(orbs)) != len(orbs):
        raise ValueError('\n\nRepeated orbital in {}.\n'.format(orbs))


def orbital_dict(orbital_dict, tags):
    '''
    Check the orbitals of *OrbitalSystem*: a list of orbitals for every tag.

    :raises TypeError: Parameter orbitals must be a dictionary.
    :raises ValueError: Every tag needs its orbitals.
    '''
    if not isinstance(orbital_dict, dict):
        raise TypeError('\n\nParameter orbitals must be a dictionary {tag: list of orbitals}.\n')
    for tag in tags:
        if tag not in orbital_dict:
            raise ValueError('\n\nParameter orbitals has no entry for tag {!r}.\n'.format(tag))
    for orbs in orbital_dict.values():
        orbitals(orbs)


def sk_params(params):
    '''
    Check a dictionary of Slater-Koster bond integrals.

    :raises TypeError: Parameter params must be a dictionary of numbers.
    :raises ValueError: Unknown bond integral.
    '''
    from tbkit.slater_koster import PARAMS
    if not isinstance(params, dict):
        raise TypeError('\n\nThe bond integrals must be a dictionary.\n')
    for key, val in params.items():
        if key not in PARAMS:
            raise ValueError('\n\nUnknown bond integral {!r}: use {}.\n'.format(key, PARAMS))
        if not isinstance(val, (int, float)):
            raise TypeError('\n\nBond integral {!r} must be a real number.\n'.format(key))


def sk_pair_params(params, tags_i, tags_j):
    '''
    Check the bond integrals of *set_slater_koster*: one dictionary, or one
    per tag pair of the bonds.

    :returns: True if keyed by tag pair.
    :raises ValueError: A tag pair of the bonds has no bond integrals.
    '''
    from tbkit.slater_koster import PARAMS
    if not isinstance(params, dict):
        raise TypeError('\n\nThe bond integrals must be a dictionary.\n')
    by_pair = bool(params) and all(k not in PARAMS for k in params)
    if not by_pair:
        sk_params(params)
        return False
    for a, b in set(zip(tags_i, tags_j)):
        if a + b not in params:
            raise ValueError('\n\nNo bond integrals for the tag pair {!r}.\n'.format(a + b))
        sk_params(params[a + b])
    return True


def bond_vector(d):
    '''
    Check a bond vector: 2 or 3 reals, nonzero.

    :raises ValueError: The bond vector must have 2 or 3 components and be nonzero.
    '''
    if d.ndim != 1 or len(d) not in (2, 3) or not np.any(d):
        raise ValueError('\n\nThe bond vector must have 2 or 3 components, not all zero.\n')


def orbital_of_site(orbital, orbs):
    '''
    Check that a site carries an orbital.

    :raises ValueError: The site has no such orbital.
    '''
    if orbital not in orbs:
        raise ValueError('\n\nThe site has no orbital {!r} (it has {}).\n'.format(orbital, orbs))


def spin_index(spin, ns):
    '''
    Check a spin index: 0, or 0/1 with spin.

    :raises ValueError: Parameter spin must be 0 (or 1 with spin).
    '''
    if spin not in range(ns):
        raise ValueError('\n\nParameter spin must be {}.\n'.format('0 or 1' if ns == 2 else '0'))


def set_onsite_orb(dict_onsite, tags, orbital_dict):
    '''
    Check *OrbitalSystem.set_onsite*: numbers, or {orbital: number} dictionaries.

    :raises TypeError: Onsite energies must be numbers or dictionaries.
    :raises ValueError: Unknown tag or orbital.
    '''
    if not isinstance(dict_onsite, dict):
        raise TypeError('\n\nParameter onsite must be a dictionary.\n')
    for tag, val in dict_onsite.items():
        if tag not in tags:
            raise ValueError('\n\nParameter onsite keys must be a tag.\n')
        if isinstance(val, dict):
            for o, e in val.items():
                orbital_of_site(o, orbital_dict[tag])
                if not isinstance(e, (int, float, complex)):
                    raise TypeError('\n\nOnsite energies must be numbers.\n')
        elif not isinstance(val, (int, float, complex)):
            raise TypeError('\n\nOnsite energies must be numbers or {orbital: number} dictionaries.\n')


def spinful(spin):
    '''
    Check that the model has spin.

    :raises ValueError: This term needs spin=True.
    '''
    if not spin:
        raise ValueError('\n\nThis term needs a spinful model (spin=True).\n')


def vector3(vec, var_name):
    '''
    Check a tuple of three real numbers.

    :raises ValueError: Parameter must be a tuple of three real numbers.
    '''
    if not isinstance(vec, (tuple, list)) or len(vec) != 3 or \
            not all(isinstance(c, (int, float)) for c in vec):
        raise ValueError('\n\nParameter {} must be a tuple of three real numbers.\n'.format(var_name))


def p_shell(tags, orbital_dict):
    '''
    Check that the tags carry the three p orbitals.

    :raises ValueError: The tags must carry px, py and pz.
    '''
    if not isinstance(tags, list) or not tags:
        raise ValueError('\n\nParameter tags must be a non-empty list of tags with p orbitals.\n')
    for tag in tags:
        if tag not in orbital_dict or not {'px', 'py', 'pz'} <= set(orbital_dict[tag]):
            raise ValueError('\n\nTag {!r} must carry the orbitals px, py and pz.\n'.format(tag))


def block_shape(mat, shape):
    '''
    Check the shape of a hopping block.

    :raises ValueError: The block must have the given shape.
    '''
    if mat.shape != shape:
        raise ValueError('\n\nThe hopping block must have shape {}.\n'.format(shape))


def orbital_terms(hop, blocks):
    '''
    Check that an *OrbitalSystem* has hoppings.

    :raises RuntimeError: Run set_hopping or set_slater_koster first.
    '''
    if not hop.size and not any(blocks.values()):
        raise RuntimeError('\n\nRun method set_hopping or set_slater_koster first.\n')


def no_left(left):
    '''
    Check that left eigenvectors are not asked for with an overlap matrix.

    :raises ValueError: left=True needs an orthogonal basis.
    '''
    if left:
        raise ValueError('\n\nleft=True needs an orthogonal basis (no overlap).\n')


def orthogonal(overlap):
    '''
    Check that the basis is orthogonal.

    :raises ValueError: This calculation needs an orthogonal basis.
    '''
    if overlap is not None:
        raise ValueError('\n\nThis calculation needs an orthogonal basis (no overlap matrix).\n')


####################################
# MEAN FIELD, SUPERCONDUCTIVITY
####################################


def hermitian_dense(ham):
    '''
    Check that a dense matrix is Hermitian.

    :raises ValueError: This calculation requires a Hermitian Hamiltonian.
    '''
    if not np.allclose(ham, ham.conj().T, atol=1e-12):
        raise ValueError('\n\nThis calculation requires a Hermitian Hamiltonian.\n')


def integer_electrons(n_electrons):
    '''
    Check that the number of electrons is an integer (zero temperature).

    :raises ValueError: At zero temperature, n_electrons must be an integer.
    '''
    if abs(n_electrons - round(n_electrons)) > 1e-12:
        raise ValueError('\n\nAt zero temperature, n_electrons must be an integer '
                                    '(use a small temperature for a fractional filling).\n')


def mixing(mixing):
    '''
    Check the density mixing: a real in (0, 1].

    :raises ValueError: Parameter mixing must be in (0, 1].
    '''
    positive_real(mixing, 'mixing')
    if mixing > 1:
        raise ValueError('\n\nParameter mixing must be in (0, 1].\n')


def pairing(delta, n):
    '''
    Check a BdG pairing matrix: (n, n), antisymmetric.

    :raises ValueError: The pairing matrix must be (n, n) and antisymmetric.
    '''
    if delta.shape != (n, n):
        raise ValueError('\n\nThe pairing matrix must have the shape ({0}, {0}) of ham.\n'.format(n))
    if (abs(delta + delta.T) > 1e-12).nnz:
        raise ValueError('\n\nThe pairing matrix must be antisymmetric (Delta^T = -Delta).\n')


def hermitian_kspace(hermitian, overlap):
    '''
    Check that a KSpace model is Hermitian, with an orthogonal basis.

    :raises ValueError: The model must be Hermitian, without overlap.
    '''
    if not hermitian or overlap:
        raise ValueError('\n\nThe model must be Hermitian, with an orthogonal basis.\n')


def pairing_kspace(pairing, norb, ndim):
    '''
    Check the pairings of *bdg.bdg_kspace*.

    :raises TypeError: Parameter pairing must be a list of dictionaries.
    :raises KeyError: "i", "j", "R" and "delta" must be keys.
    :raises ValueError: Orbital indices below norb; R a tuple of ndim integers.
    '''
    if not isinstance(pairing, list):
        raise TypeError('\n\nParameter pairing must be a list of dictionaries.\n')
    for dic in pairing:
        if not isinstance(dic, dict):
            raise TypeError('\n\nParameter pairing must be a list of dictionaries.\n')
        if not {'i', 'j', 'R', 'delta'} <= set(dic):
            raise KeyError('\n\n"i", "j", "R" and "delta" must be dictionary keys.\n')
        if not all(isinstance(dic[k], int) and 0 <= dic[k] < norb for k in ('i', 'j')):
            raise ValueError('\n\n"i" and "j" must be orbital indices between 0 and {}.\n'.format(norb - 1))
        if not isinstance(dic['R'], tuple) or len(dic['R']) != ndim or \
                not all(isinstance(n, int) for n in dic['R']):
            raise ValueError('\n\n"R" must be a tuple of {} integers.\n'.format(ndim))
        number(dic['delta'], '"delta"')


####################################
# FLOQUET
####################################


def positive_int_zero(var, var_name):
    '''
    Check if *var* is a positive integer or zero.

    :raises TypeError: Parameter *var* must be an integer.
    :raises ValueError: Parameter *var* must be a positive integer or zero.
    '''
    if not isinstance(var, int):
        raise TypeError('\n\nParameter {} must be an integer.\n'.format(var_name))
    if var < 0:
        raise ValueError('\n\nParameter {} must be a positive integer or zero.\n'.format(var_name))


def harmonics(harm):
    '''
    Check the harmonics of *floquet.sambe_hamiltonian*: a dictionary with (at least) m = 0.

    :raises TypeError: The harmonics must be a dictionary {m: matrix}.
    :raises ValueError: The harmonics must contain m = 0 and square matrices of one shape.
    '''
    if not isinstance(harm, dict):
        raise TypeError('\n\nThe harmonics must be a dictionary {m: matrix}.\n')
    if 0 not in harm:
        raise ValueError('\n\nThe harmonics must contain the static part, m = 0.\n')
    shapes = {np.shape(h) for h in harm.values()}
    if len(shapes) != 1 or len(shapes.pop()) != 2:
        raise ValueError('\n\nThe harmonics must be square matrices of one shape.\n')


def kspace(ks, cls):
    '''
    Check that *ks* is a KSpace instance.

    :raises TypeError: Parameter ks must be a KSpace instance.
    '''
    if not isinstance(ks, cls):
        raise TypeError('\n\nParameter ks must be a KSpace instance.\n')


def not_floquet():
    '''
    :raises ValueError: Not available for a Floquet (driven) model.
    '''
    raise ValueError('\n\nThis method needs the static hoppings: it is not available '
                                'for a Floquet (driven) model.\n')


def sk_bonds(bonds):
    '''
    Check the bonds of *slater_koster.sk_kspace*: {n: bond integrals}.

    :raises TypeError: Parameter bonds must be a dictionary {n: bond integrals}.
    '''
    if not isinstance(bonds, dict) or not all(isinstance(n, int) and n > 0 for n in bonds):
        raise TypeError('\n\nParameter bonds must be a dictionary {n: bond integrals}, '
                                  'n a positive integer.\n')
    for params in bonds.values():
        sk_params(params)


####################################
# HALL CONDUCTIVITIES
####################################


def fermi_energies(e_fermi):
    '''
    Check the Fermi energies of the Hall conductivities: a real number, or a
    non-empty array of finite real numbers.

    :raises TypeError: Parameter e_fermi must be a real number or an array of real numbers.
    :raises ValueError: Parameter e_fermi must be non-empty and finite.
    '''
    arr = np.asarray(e_fermi)
    if arr.dtype.kind not in 'iuf':
        raise TypeError('\n\nParameter e_fermi must be a real number or an array '
                                  'of real numbers.\n')
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError('\n\nParameter e_fermi must contain at least one finite number.\n')


def hermitian_model(hermitian):
    '''
    Check that a KSpace model is Hermitian (a Kubo conductivity needs real
    bands and orthonormal eigenstates).

    :raises ValueError: This calculation requires a Hermitian model.
    '''
    if not hermitian:
        raise ValueError('\n\nThis calculation requires a Hermitian model (no gain/loss, '
                                    'no hermitian=False hopping).\n')


def k_fixed_hall(k_fixed, dim, full_allowed=True):
    '''
    Check parameter *k_fixed* of the Hall conductivities: None (the whole
    Brillouin zone) or a real number (a plane of a 3D model).

    :raises TypeError: Parameter k_fixed must be None or a real number.
    :raises ValueError: A 3D spin Hall conductivity needs a plane (k_fixed).
    '''
    if k_fixed is not None:
        real_number(k_fixed, 'k_fixed')
    elif dim == 3 and not full_allowed:
        raise ValueError('\n\nIn 3D, give the plane of the spin Hall conductivity: '
                                    'k_fixed must be a real number.\n')


def spin_axis(axis):
    '''
    Check the spin quantization axis: 'x', 'y' or 'z'.

    :raises ValueError: Parameter spin_axis must be "x", "y" or "z".
    '''
    if axis not in ('x', 'y', 'z'):
        raise ValueError('\n\nParameter spin_axis must be "x", "y" or "z".\n')


def refine_fraction(fraction):
    '''
    Check the fraction of mesh points refined by the adaptive mesh: a real in (0, 1].

    :raises TypeError: Parameter refine_fraction must be a real number.
    :raises ValueError: Parameter refine_fraction must be in (0, 1].
    '''
    positive_real(fraction, 'refine_fraction')
    if fraction > 1:
        raise ValueError('\n\nParameter refine_fraction must be in (0, 1].\n')


def velocity(vel, n, var_name):
    '''
    Check a velocity operator: a square (sparse or dense) matrix of the
    Hamiltonian's shape.

    :raises ValueError: The velocity must be a (n, n) matrix.
    '''
    if getattr(vel, 'shape', None) != (n, n):
        raise ValueError('\n\nParameter {} must be a ({}, {}) matrix, like ham.\n'.format(var_name, n, n))


def positive_overlap(s_min):
    '''
    Check that the overlap matrix S(k) is positive definite.

    :raises ValueError: The overlap matrix must be positive definite.
    '''
    if s_min <= 0:
        raise ValueError('\n\nThe overlap matrix S(k) must be positive definite '
                                    '(smallest eigenvalue {:.3g}).\n'.format(s_min))


####################################
# FLOQUET: STEP DRIVES, WINDING NUMBER
####################################


def branch_cut(epsilon):
    '''
    Check the branch cut of a Floquet Hamiltonian: None or a real number.

    :raises TypeError: Parameter epsilon must be None or a real number.
    '''
    if epsilon is not None and (isinstance(epsilon, bool)
                                        or not isinstance(epsilon, (int, float, np.integer, np.floating))):
        raise TypeError('\n\nParameter epsilon must be None or a real number.\n')


def durations(durations, n):
    '''
    Check the step durations of a step drive: n positive reals.

    :raises TypeError: Parameter durations must be a list of real numbers.
    :raises ValueError: Parameter durations must hold one positive number per step.
    '''
    if not isinstance(durations, (list, tuple, np.ndarray)):
        raise TypeError('\n\nParameter durations must be a list of real numbers.\n')
    if len(durations) != n:
        raise ValueError('\n\nParameter durations must hold one duration per step ({}).\n'.format(n))
    for d in durations:
        if isinstance(d, bool) or not isinstance(d, (int, float, np.integer, np.floating)):
            raise TypeError('\n\nParameter durations must be a list of real numbers.\n')
        if d <= 0:
            raise ValueError('\n\nParameter durations must be positive numbers.\n')


def step_models(models, kspace_cls, system_cls):
    '''
    Check the models of a step drive: a non-empty list, either of KSpace
    instances (Hermitian, orthogonal basis, one lattice size) or of System
    instances and square matrices. Returns 'kspace' or 'matrix'.

    :raises TypeError: Parameter models must be a non-empty list of KSpace, System or matrices.
    :raises ValueError: The KSpace models must be Hermitian, with no overlap, of one size.
    '''
    if not isinstance(models, (list, tuple)) or len(models) == 0:
        raise TypeError('\n\nParameter models must be a non-empty list of KSpace instances, '
                                'or of System instances and matrices.\n')
    if all(isinstance(m, kspace_cls) for m in models):
        for m in models:
            if (m.norb, m.dim) != (models[0].norb, models[0].dim):
                raise ValueError('\n\nThe KSpace models must have the same orbitals and dimension.\n')
            if not m.is_hermitian():
                raise ValueError('\n\nThe KSpace models must be Hermitian.\n')
            if m._overlap_hop:
                raise ValueError('\n\nThe KSpace models need an orthogonal basis (no overlap).\n')
        return 'kspace'
    for m in models:
        if not isinstance(m, system_cls) and getattr(m, 'ndim', None) != 2:
            raise TypeError('\n\nParameter models must be a list of KSpace instances only, '
                                    'or of System instances and square matrices.\n')
    return 'matrix'


def step_matrices(hams):
    '''
    Check the Hamiltonians of a real-space step drive: Hermitian square
    matrices of one shape.

    :raises TypeError: Parameter hams must be a non-empty list of matrices.
    :raises ValueError: The matrices must be square, Hermitian and of one shape.
    '''
    if not isinstance(hams, (list, tuple)) or len(hams) == 0:
        raise TypeError('\n\nParameter hams must be a non-empty list of matrices.\n')
    shape = np.shape(hams[0])
    for h in hams:
        h = np.asarray(h)
        if h.ndim != 2 or h.shape[0] != h.shape[1] or h.shape != shape:
            raise ValueError('\n\nThe step Hamiltonians must be square matrices of one shape.\n')
        hermitian_dense(h)


def drive_matrix(ham, norb):
    '''
    Check the output of a Bloch drive ham_kt(k, t): a Hermitian (norb, norb) matrix.

    :raises ValueError: ham_kt(k, t) must return a Hermitian (norb, norb) matrix.
    '''
    if np.shape(ham) != (norb, norb):
        raise ValueError('\n\nham_kt(k, t) must return a ({}, {}) matrix.\n'.format(norb, norb))
    hermitian_dense(np.asarray(ham))


def time_in_period(t, period):
    '''
    Check a time within the period: None or a real in [0, period].

    :raises TypeError: Parameter t must be None or a real number.
    :raises ValueError: Parameter t must be in [0, period].
    '''
    if t is None:
        return
    real_number(t, 't')
    if not 0 <= t <= period:
        raise ValueError('\n\nParameter t must be in [0, {}].\n'.format(period))


def nk_min(nk, n_min):
    '''
    Check that every k-mesh size is at least n_min.

    :raises ValueError: Parameter nk must be at least n_min.
    '''
    if min(nk) < n_min:
        raise ValueError('\n\nParameter nk must be at least {}.\n'.format(n_min))
