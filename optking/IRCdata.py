### Class to store points on the IRC
import logging
import os

import numpy as np

from .exceptions import OptError
from .printTools import print_geom_string, print_mat_string


class IRCpoint(object):
    """Holds data for one step on the IRC.
    Parameters
    ----------
    step_number : int
        step number
    q_pivot :    ndarray
        pivot point for next step
    x_pivot :    ndarray
        pivot point for next step; save so that q_pivot can be recomputed if desired
    q       :    ndarray
        internal coordinate values
    x       :    ndarray
        cartesian coordinate values
    f_q     :    ndarray
        internal coordinate forces
    f_x     :    ndarray
        cartesian coordinate forces
    energy  :    float
        total energy
    step_dist :  float
    arc_dist  :  float
    line_dist :  float
    """

    def __init__(
        self,
        step_number,
        q,
        x,
        f_q,
        f_x,
        energy,
        q_pivot,
        x_pivot,
        step_dist,
        arc_dist,
        line_dist,
    ):
        self.step_number = step_number
        self.q = q
        self.x = x
        self.f_q = f_q
        self.f_x = f_x
        self.energy = energy
        self.q_pivot = q_pivot
        self.x_pivot = x_pivot
        self.step_dist = step_dist
        self.arc_dist = arc_dist
        self.line_dist = line_dist

    def add_pivot(self, q_p, x_p):
        self.q_pivot = q_p
        self.x_pivot = x_p

    def dict_output(self):
        s = {
            "Step Number": self.step_number,
            "Intco Values": self.q,
            "Geometry": self.x,
            "Internal Forces": self.f_q,
            "Cartesian Forces": self.f_x,
            "Energy": self.energy,
            "Pivot Intco Values": self.q_pivot,
            "Pivot Geometry": self.x_pivot,
            "Step Distance": self.step_dist,
            "Arc Distance": self.arc_dist,
            "Line Distance": self.line_dist,
        }
        return s


class IRCdata(object):
    """Stores obtained points along the IRC as well as information about
    the status of the IRC computation"""

    __step_size = 0.0
    __direction = None
    __running_step_dist = 0.0
    __running_arc_dist = 0.0
    __running_line_dist = 0.0

    def __init__(self):
        self.go = True
        self.irc_points = []
        self.atom_symbols = None

    def set_atom_symbols(self, atom_symbols):  # just for printing
        self.atom_symbols = atom_symbols.copy()  # just for printing

    def set_step_size_and_direction(self, step_size, direction):
        self.__step_size = step_size
        self.__direction = direction

    def add_irc_point(self, step_number, q_in, x_in, f_q, f_x, E, lineDistStep=0, arcDistStep=0):
        if len(self.irc_points) != 0:
            if self.__direction == "FORWARD":
                sign = 1
            elif self.__direction == "BACKWARD":
                sign = -1
                step_number *= -1
            else:
                raise OptError("IRC direction must be set to FORWARD or BACKWARD")
            # step_dist = sum of all steps to and from pivot points, a multiple of
            # the step_size
            self.__running_step_dist += sign * self.__step_size
            # line distance is sum of all steps directly between rxnpath points, ignoring
            # pivot points
            self.__running_line_dist += sign * lineDistStep
            # distance along a circular arc connecting rxnpath points
            self.__running_arc_dist += sign * arcDistStep

        onepoint = IRCpoint(
            step_number,
            q_in,
            x_in,
            f_q,
            f_x,
            E,
            None,
            None,
            self.__running_step_dist,
            self.__running_arc_dist,
            self.__running_line_dist,
        )
        self.irc_points.append(onepoint)

        logger = logging.getLogger(__name__)
        pindex = len(self.irc_points) - 1
        outstr = "\nAdding IRC point %d\n" % pindex
        outstr += print_geom_string(self.atom_symbols, x_in, "Angstroms")
        logger.info(outstr)

    def add_pivot_point(self, q_p, x_p, step=None):
        index = -1 if step is None else step
        logger = logging.getLogger(__name__)
        pindex = (len(self.irc_points) - 1) if step is None else step
        logger.debug("Adding pivot point (index %d) for finding rxnpath point %d" % (pindex, pindex + 1))
        self.irc_points[index].add_pivot(q_p, x_p)

    # Return most recent IRC step data unless otherwise specified
    def step_number(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].step_number

    @property
    def step_size(self):
        return self.__step_size

    def current_step_number(self):
        return len(self.irc_points)

    def q_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q_pivot

    def x_pivot(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].x_pivot

    def q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].q

    def x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].x

    def f_q(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_q

    def f_x(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].f_x

    def energy(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].energy

    def line_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].line_dist

    def arc_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].arc_dist

    def step_dist(self, step=None):
        index = -1 if step is None else step
        return self.irc_points[index].step_dist

    def test_for_irc_minimum(self, f_q):
        """ Given current forces, checks if we are at/near a minimum
        For now, checks if forces are opposite those are previous pivot point
        """

        unit_f = f_q / np.linalg.norm(f_q)  # current forces
        f_rxn = self.f_q()  # forces at most recent rxnpath point
        unit_f_rxn = f_rxn / np.linalg.norm(f_rxn)
        overlap = np.dot(unit_f, unit_f_rxn)

        logger = logging.getLogger(__name__)
        logger.info("Overlap of forces with previous rxnpath point %8.4f" % overlap)

        if overlap < -0.7:
            return True

        # TODO  Look at line distance criterion when distances are working.
        # elif:
        #    g_line_dist(p_irc_data->size()-1) - g_line_dist(p_irc_data->size()-2)) < s*10e-03)
        #    return True

        return False

    def progress_report(self):
        blocks = 4  # TODO: make dynamic
        sign = 1
        Ncoord = len(self.q())

        # logging.basicConfig(filename='ircprogress.log',level=logging.DEBUG)

        irc_report = os.path.join(os.getcwd(), "ircprogress.log")  # prepare progress report
        with open(irc_report, "w") as irc_prog:
            irc_prog.truncate(0)

        irc_log = logging.getLogger("ircprogress")
        irc_handle = logging.FileHandler(os.path.join(os.getcwd(), "ircprogress.log"), "w")
        irc_handle.setLevel(logging.DEBUG)
        irc_log.addHandler(irc_handle)

        out = "\n"
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC            ****      IRC Report      ****\n"
        out += "@IRC ----------------------------------------------\n"
        out += "@IRC  Step    Energy              Change in Energy \n"
        out += "@IRC ----------------------------------------------\n"
        for i in range(len(self.irc_points)):
            if i == 0:
                DE = self.energy(i)
            else:
                DE = self.energy(i) - self.energy(i - 1)
            out += "@IRC  %3d %18.12lf  %18.12lf\n" % (i, self.energy(i), DE)
        out += "@IRC ----------------------------------------------\n\n"

        # Print Internal Coordinates for Each step
        out += "@IRC -----------------------------------------------------\n"
        out += "@IRC              ****     IRC Steps     ****             \n"
        out += "@IRC -----------------------------------------------------"
        for j in range(Ncoord // blocks):
            out += "\n@IRC        |          Distance         |\n"
            out += "@IRC Step   | Step    Arc       Line    |"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "-------------"
            out += "\n"
            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (
                    i,
                    sign * self.step_dist(i),
                    sign * self.arc_dist(i),
                    sign * self.line_dist(i),
                )
                for k in range(j * blocks, (j + 1) * blocks):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"

            out += "@IRC --------------------------------------"
            for i in range(j * blocks, (j + 1) * blocks):
                out += "-------------"
        if Ncoord % blocks != 0:
            out += "\n@IRC         |          Distance         |\n"
            out += "@IRC  Step   | Step    Arc       Line    |"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "    Coord %3d" % i
            out += "\n"
            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"
            out += "\n"

            for i in range(len(self.irc_points)):
                out += "@IRC  %3d %9.2lf %9.5lf  %9.5lf   " % (
                    i,
                    sign * self.step_dist(i),
                    sign * self.arc_dist(i),
                    sign * self.line_dist(i),
                )
                for k in range(Ncoord - (Ncoord % blocks), Ncoord):
                    out += "%13.8f" % self.q(i)[k]
                out += "\n"

            out += "@IRC --------------------------------------"

            for i in range(Ncoord - (Ncoord % blocks), Ncoord):
                out += "-------------"

        out += "\n"
        out += "\n"
        irc_log.info(out)

        # out += mol.print_coords(psi_outfile, qc_outfile)
        # out += mol.print_simples(psi_outfile, qc_outfile)

    def rxnpath_dict(self):
        rp = [self.irc_points[i].dict_output() for i in range(len(self.irc_points))]
        return rp


history = 0
