import builtins
from pprint import pprint
import sys

builtins.pprint = pprint
builtins.stdout = sys.stdout
