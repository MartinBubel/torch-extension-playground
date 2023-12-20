import torch  # required as otherwise: ImportError: DLL load failed while importing lltm_cpp: The specified module could not be found.
import lltm_cpp


print(lltm_cpp.forward)
print(help(lltm_cpp.forward))
