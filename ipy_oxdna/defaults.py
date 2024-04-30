# I'm not sure this class needs to exist
import json
import re
from typing import Union
from importlib import resources

class DefaultInput:
    _name: str
    _input: dict[str, str]

    def __init__(self, name: str):
        self._name = name
        self.reset()

    def reset(self):
        """
        reloads input dict from file, clearing evaluated info
        """
        data_path = resources.files('ipy_oxdna') / "defaults" / "inputs" / f"{self._name}.json"
        with data_path.open("r") as f:
            data = json.load(f)
        self._input = data

    def evaluate(self, **kwargs: dict[str, Union[float, int, str, bool]]):
        """
        evaluates dynamic values in the default dict, using the info provided
        in keyword arguements
        """
        r = r'f\(([^)]+)\) = (.+)'
        for key in self._input:
            # if the value is an expression that needs evaluation
            if isinstance(self._input[key], str):
                match = re.match(r, self._input[key])
                if match:
                    values_str, expression = match.groups()
                    # Split the values by comma and strip whitespace
                    argnames = [value.strip() for value in values_str.split(',')]
                    if not all([argname in kwargs for argname in argnames]):
                        missing_arg = [argname for argname in argnames if argname not in kwargs][0]
                        raise MissingParamError(key, missing_arg, list(kwargs.keys()))
                    expression_eval = expression
                    for argname in argnames:
                        expression_eval = expression_eval.replace(argname, f"{kwargs[argname]}")
                    try:
                        self._input[key] = eval(expression)
                    except SyntaxError as e:
                        raise ValueError(f"Cannot parse expression {expression}")
            # if it's not an expression we can just skip

    def get_dict(self) -> dict[str, str]:
        """
        Returns: the values
        """
        # verify that all params are valid
        r = r'f\(([^)]+)\) = (.+)'
        for key in self._input:
            if isinstance(self._input[key], str):
                # if the value is an expression that needs evaluation
                match = re.match(r, self._input[key])
                if match:
                    raise IncompleteInputError(key)
        return {
            key: str(self._input[key]) for key in self._input
        }

    def __getitem__(self, item: str) -> str:
        return str(self._input[item])


def get_default_input(name: str) -> DefaultInput:
    return DefaultInput(name)


# todo: better
SEQ_DEP_PARAMS: dict[str, float] = {
    "STCK_FACT_EPS": 0.18,
    "STCK_G_C": 1.69339,
    "STCK_C_G": 1.74669,
    "STCK_G_G": 1.61295,
    "STCK_C_C": 1.61295,
    "STCK_G_A": 1.59887,
    "STCK_T_C": 1.59887,
    "STCK_A_G": 1.61898,
    "STCK_C_T": 1.61898,
    "STCK_T_G": 1.66322,
    "STCK_C_A": 1.66322,
    "STCK_G_T": 1.68032,
    "STCK_A_C": 1.68032,
    "STCK_A_T": 1.56166,
    "STCK_T_A": 1.64311,
    "STCK_A_A": 1.84642,
    "STCK_T_T": 1.58952,
    "HYDR_A_T": 0.88537,
    "HYDR_T_A": 0.88537,
    "HYDR_C_G": 1.23238,
    "HYDR_G_C": 1.23238
}

NA_PARAMETERS = {
    "HYDR_A_U": 1.21,
    "HYDR_A_T": 1.37,
    "HYDR_rC_dG": 1.61,
    "HYDR_rG_dC": 1.77
}

RNA_PARAMETERS = {
    "HYDR_A_T": 0.820419,
    "HYDR_C_G": 1.06444,
    "HYDR_G_T": 0.510558,
    "STCK_G_C": 1.27562,
    "STCK_C_G": 1.60302,
    "STCK_G_G": 1.49422,
    "STCK_C_C": 1.47301,
    "STCK_G_A": 1.62114,
    "STCK_T_C": 1.16724,
    "STCK_A_G": 1.39374,
    "STCK_C_T": 1.47145,
    "STCK_T_G": 1.28576,
    "STCK_C_A": 1.58294,
    "STCK_G_T": 1.57119,
    "STCK_A_C": 1.21041,
    "STCK_A_T": 1.38529,
    "STCK_T_A": 1.24573,
    "STCK_A_A": 1.31585,
    "STCK_T_T": 1.17518,
    "CROSS_A_A": 59.9626,
    "CROSS_A_T": 59.9626,
    "CROSS_T_A": 59.9626,
    "CROSS_A_C": 59.9626,
    "CROSS_C_A": 59.9626,
    "CROSS_A_G": 59.9626,
    "CROSS_G_A": 59.9626,
    "CROSS_G_G": 59.9626,
    "CROSS_G_C": 59.9626,
    "CROSS_C_G": 59.9626,
    "CROSS_G_T": 59.9626,
    "CROSS_T_G": 59.9626,
    "CROSS_C_C": 59.9626,
    "CROSS_C_T": 59.9626,
    "CROSS_T_C": 59.9626,
    "CROSS_T_T": 59.9626,
    "ST_T_DEP": 1.97561
}


class IncompleteInputError(Exception):
    objectionable_input_key: str

    def __init__(self, k: str):
        self.objectionable_input_key = k

    def __str__(self) -> str:
        return f"Input file hasn't been evaluated for input key {self.objectionable_input_key}"


class MissingParamError(Exception):
    key: str
    missing: str
    provided_params: list[str]

    def __init__(self, key: str, missing: str, provided_params: list[str]):
        self.key = key
        self.missing = missing
        self.provided_params = provided_params

    def __str__(self) -> str:
        return f"Cannot evaluate dynamic expression for `{self.key}`, missing parameter `{self.missing}`. " \
               f"Provided vals for {', '.join(self.provided_params)}"
