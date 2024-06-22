import pandas as pd

import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt

# for feature reduction
#ga_sli = [[0.582054309, 0.564344746, 0.592680047, 0.572609209, 0.557260921, 0.60094451, 0.585596222, 0.574970484, 0.59149941, 0.564344746, 0.587957497, 0.565525384, 0.60094451, 0.58677686, 0.576151122], [0.570247934, 0.514757969, 0.574970484, 0.576151122, 0.572609209, 0.564344746, 0.556080283, 0.547815821, 0.536009445, 0.564344746, 0.558441558, 0.552538371, 0.556080283, 0.563164109, 0.539551358], [0.520661157, 0.526564345, 0.52420307, 0.561983471, 0.53837072, 0.557260921, 0.520661157, 0.563164109, 0.52420307, 0.521841795, 0.520661157, 0.520661157, 0.52420307, 0.518299882, 0.552538371], [0.510035419, 0.534828808, 0.501770956, 0.511216057, 0.536009445, 0.497048406, 0.506493506, 0.527744982, 0.517119244, 0.536009445, 0.526564345, 0.521841795, 0.514757969, 0.53837072, 0.512396694], [0.518299882, 0.497048406, 0.514757969, 0.534828808, 0.525383707, 0.492325856, 0.519480519, 0.491145218, 0.52892562, 0.504132231, 0.507674144, 0.507674144, 0.514757969, 0.480519481, 0.47107438], [0.507674144, 0.502951594, 0.519480519, 0.501770956, 0.526564345, 0.510035419, 0.523022432, 0.485242031, 0.511216057, 0.501770956, 0.488783943, 0.519480519, 0.547815821, 0.506493506, 0.530106257], [0.495867769, 0.514757969, 0.510035419, 0.525383707, 0.493506494, 0.530106257, 0.527744982, 0.526564345, 0.473435655, 0.486422668, 0.473435655, 0.494687131, 0.508854782, 0.520661157, 0.52420307]]
#svd_woa = [[0.552538371, 0.537190083, 0.53364817, 0.554899646, 0.527744982, 0.52892562, 0.530106257, 0.557260921, 0.52892562, 0.53364817, 0.547815821, 0.544273908, 0.561983471, 0.532467532, 0.546635183], [0.517119244, 0.518299882, 0.546635183, 0.534828808, 0.53837072, 0.52892562, 0.534828808, 0.53837072, 0.525383707, 0.53364817, 0.519480519, 0.534828808, 0.510035419, 0.564344746, 0.544273908], [0.492325856, 0.514757969, 0.532467532, 0.517119244, 0.507674144, 0.507674144, 0.488783943, 0.499409681, 0.514757969, 0.520661157, 0.491145218, 0.494687131, 0.510035419, 0.510035419, 0.507674144], [0.454545455, 0.526564345, 0.469893743, 0.506493506, 0.492325856, 0.489964581, 0.515938607, 0.493506494, 0.499409681, 0.502951594, 0.506493506, 0.504132231, 0.492325856, 0.487603306, 0.510035419], [0.505312869, 0.517119244, 0.480519481, 0.497048406, 0.530106257, 0.507674144, 0.511216057, 0.494687131, 0.495867769, 0.504132231, 0.532467532, 0.495867769, 0.515938607, 0.497048406, 0.479338843], [0.494687131, 0.497048406, 0.47107438, 0.523022432, 0.497048406, 0.494687131, 0.527744982, 0.494687131, 0.523022432, 0.495867769, 0.491145218, 0.499409681, 0.47107438, 0.514757969, 0.507674144], [0.52420307, 0.499409681, 0.482880756, 0.494687131, 0.500590319, 0.527744982, 0.508854782, 0.506493506, 0.487603306, 0.498229044, 0.486422668, 0.484061393, 0.500590319, 0.504132231, 0.47579693]]
#voice_woa = [[0.559622196, 0.558441558, 0.564344746, 0.554899646, 0.553719008, 0.564344746, 0.551357733, 0.556080283, 0.552538371, 0.551357733, 0.54309327, 0.550177096, 0.553719008, 0.558441558, 0.550177096], [0.54309327, 0.547815821, 0.559622196, 0.544273908, 0.559622196, 0.567886659, 0.547815821, 0.541912633, 0.559622196, 0.548996458, 0.552538371, 0.547815821, 0.553719008, 0.570247934, 0.551357733], [0.52892562, 0.544273908, 0.53364817, 0.556080283, 0.511216057, 0.539551358, 0.531286895, 0.532467532, 0.54309327, 0.52892562, 0.525383707, 0.534828808, 0.530106257, 0.523022432, 0.510035419], [0.53364817, 0.513577332, 0.539551358, 0.544273908, 0.564344746, 0.531286895, 0.505312869, 0.532467532, 0.523022432, 0.494687131, 0.513577332, 0.556080283, 0.52892562, 0.514757969, 0.525383707], [0.53364817, 0.495867769, 0.508854782, 0.518299882, 0.502951594, 0.507674144, 0.521841795, 0.517119244, 0.47579693, 0.521841795, 0.488783943, 0.507674144, 0.52892562, 0.493506494, 0.521841795], [0.498229044, 0.493506494, 0.512396694, 0.532467532, 0.53837072, 0.508854782, 0.518299882, 0.514757969, 0.513577332, 0.534828808, 0.482880756, 0.501770956, 0.510035419, 0.512396694, 0.505312869], [0.505312869, 0.497048406, 0.47107438, 0.532467532, 0.486422668, 0.518299882, 0.508854782, 0.506493506, 0.497048406, 0.495867769, 0.513577332, 0.465171192, 0.506493506, 0.497048406, 0.486422668]]


# for f1-score

ga_sli = [[0.96325903, 0.960237894, 0.956576757, 0.967076221, 0.9651363, 0.970472194, 0.966998335, 0.968845566, 0.962097831, 0.961922358, 0.967228814, 0.9620107, 0.960144899, 0.956780319, 0.948196409], [0.963437202, 0.961556604, 0.968917845, 0.967303545, 0.956978349, 0.9620107, 0.965383962, 0.958460754, 0.958557127, 0.960329617, 0.966998335, 0.968989135, 0.965383962, 0.9620107, 0.95119082], [0.953102897, 0.95647286, 0.963524476, 0.956780319, 0.968772285, 0.952644703, 0.970401545, 0.942681319, 0.959955031, 0.965464272, 0.956780319, 0.949865171, 0.949980821, 0.965464272, 0.945629104], [0.970472194, 0.967153043, 0.946026346, 0.9620107, 0.961922358, 0.957958539, 0.960329617, 0.956880018, 0.949386335, 0.965383962, 0.962097831, 0.967153043, 0.965219989, 0.959660303, 0.955408249], [0.961832792, 0.938613504, 0.926244344, 0.961741988, 0.925312021, 0.953430419, 0.946026346, 0.920876987, 0.86910259, 0.916123499, 0.907745222, 0.951307984, 0.871926666, 0.917970956, 0.947829983], [0.958557127, 0.95119082, 0.94926247, 0.953430419, 0.96361056, 0.949136904, 0.945629104, 0.9620107, 0.95647286, 0.958745927, 0.953213577, 0.945629104, 0.941680784, 0.960329617, 0.968468904], [0.970541882, 0.97032992, 0.949386335, 0.967228814, 0.967228814, 0.968845566, 0.953641355, 0.949386335, 0.958363046, 0.946155178, 0.958557127, 0.9548933, 0.967076221, 0.956978349, 0.962097831]]
svd_woa = [[0.682437851, 0.695238095, 0.694823393, 0.608660205, 0.688111664, 0.726247916, 0.660219342, 0.699326448, 0.730553328, 0.695099819, 0.652236136, 0.690876565, 0.663440648, 0.703795453, 0.677014752], [0.72899562, 0.729516995, 0.700536673, 0.66319166, 0.700536673, 0.689648095, 0.695303968, 0.699326448, 0.711307137, 0.701684042, 0.663224494, 0.699326448, 0.705938804, 0.696175824, 0.638449349], [0.709977768, 0.712132353, 0.738123035, 0.703557045, 0.712032539, 0.729516995, 0.651889683, 0.717647059, 0.66243314, 0.697431907, 0.744504207, 0.677014752, 0.685420448, 0.685420448, 0.716642579], [0.715580155, 0.738123035, 0.71019678, 0.734862856, 0.695099819, 0.692060946, 0.729804464, 0.669790628, 0.698051948, 0.741122159, 0.722905423, 0.715580155, 0.706495154, 0.682437851, 0.686656103], [0.737012987, 0.709352977, 0.711307137, 0.763786908, 0.71019678, 0.726062143, 0.696297454, 0.700637531, 0.690876565, 0.696711659, 0.71019678, 0.696297454, 0.71445856, 0.712357955, 0.732715369], [0.704779189, 0.722905423, 0.689627301, 0.705938804, 0.682747712, 0.701684042, 0.737673063, 0.723809524, 0.71859474, 0.709025595, 0.691113381, 0.71072523, 0.716278648, 0.71445856, 0.719205419], [0.68782657, 0.718724742, 0.721794148, 0.706495154, 0.728423889, 0.713276487, 0.739177102, 0.726247916, 0.714961776, 0.713276487, 0.713276487, 0.738123035, 0.707037243, 0.729516995, 0.712357955]]
voice_woa = [[0.416666667, 0.416666667, 0.391304348, 0.408450704, 0.416666667, 0.408450704, 0.408450704, 0.416666667, 0.416666667, 0.416666667, 0.416666667, 0.416666667, 0.416666667, 0.408450704, 0.416666667], [0.485714286, 0.416666667, 0.416666667, 0.408450704, 0.391304348, 0.416666667, 0.485714286, 0.416666667, 0.416666667, 0.416666667, 0.408450704, 0.416666667, 0.416666667, 0.416666667, 0.416666667], [0.408450704, 0.416666667, 0.416666667, 0.416666667, 0.485714286, 0.408450704, 0.4, 0.408450704, 0.408450704, 0.408450704, 0.416666667, 0.416666667, 0.416666667, 0.408450704, 0.408450704], [0.408450704, 0.485714286, 0.408450704, 0.408450704, 0.4, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.4, 0.416666667, 0.408450704, 0.416666667], [0.416666667, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.416666667, 0.408450704, 0.4, 0.416666667, 0.416666667, 0.416666667, 0.416666667, 0.416666667, 0.408450704, 0.408450704], [0.408450704, 0.408450704, 0.416666667, 0.408450704, 0.408450704, 0.408450704, 0.416666667, 0.4, 0.408450704, 0.416666667, 0.391304348, 0.416666667, 0.408450704, 0.408450704, 0.408450704], [0.416666667, 0.408450704, 0.416666667, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.416666667, 0.408450704, 0.408450704, 0.408450704, 0.408450704, 0.416666667]]



#for fitness:


#ga_sli = [[0.398478437, 0.385006365, 0.385006365, 0.372570127, 0.372570127, 0.370928013, 0.367151133, 0.36593338, 0.364938344, 0.363898409, 0.362320837, 0.362320837, 0.358686395, 0.358686395, 0.356736314, 0.356736314, 0.355132142, 0.352930718, 0.350354333, 0.350354333, 0.350354333, 0.348172547, 0.347428547, 0.347428547, 0.345852653, 0.345852653, 0.345576272, 0.34480807, 0.34480807, 0.340636979], [0.291130538, 0.276702735, 0.265066591, 0.254541529, 0.254541529, 0.251491051, 0.24840912, 0.24840912, 0.24840912, 0.24840912, 0.24840912, 0.247504387, 0.247504387, 0.244822971, 0.244822971, 0.244822971, 0.244822971, 0.244822971, 0.244822971, 0.244822971, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791], [0.159752494, 0.139631, 0.133888047, 0.132377724, 0.130425691, 0.127653684, 0.121674403, 0.121674403, 0.121557718, 0.119818624, 0.119818624, 0.119035725, 0.119035725, 0.119035725, 0.119035725, 0.118456333, 0.11844567, 0.117088114, 0.117088114, 0.117088114, 0.117088114, 0.117088114, 0.117088114, 0.116861448, 0.11684649, 0.11684649, 0.11684649, 0.116778721, 0.116210705, 0.116162117], [0.123954867, 0.117087112, 0.117087112, 0.109586906, 0.109586906, 0.109586906, 0.102905123, 0.102905123, 0.097963727, 0.097963727, 0.094630824, 0.094450182, 0.094450182, 0.094450182, 0.094450182, 0.09435127, 0.094308561, 0.094308561, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534], [0.092587601, 0.089459824, 0.087585685, 0.085496557, 0.079120349, 0.079120349, 0.079120349, 0.079120349, 0.079120349, 0.077652248, 0.077652248, 0.077652248, 0.077652248, 0.077652248, 0.074995365, 0.074995365, 0.074995365, 0.074995365, 0.074995365, 0.074995365, 0.074328819, 0.074328819, 0.074328819, 0.073827859, 0.073827859, 0.073827859, 0.073827859, 0.073827859, 0.073827859, 0.073827859], [0.090053852, 0.08575038, 0.079230846, 0.066765853, 0.056776105, 0.056776105, 0.056776105, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055000509, 0.054552806, 0.054552806, 0.053157944, 0.052799949, 0.052799949], [0.073075102, 0.073075102, 0.05047963, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.046021856, 0.046021856, 0.046021856, 0.046021856, 0.044582189, 0.044582189, 0.044582189, 0.044582189, 0.034064057, 0.034064057, 0.032788502, 0.032788502, 0.032788502, 0.032788502, 0.032788502, 0.028479114, 0.028479114, 0.026367306]]
#svd_woa = [[0.488344873, 0.462057473, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394], [0.434280413, 0.42963652, 0.429252614, 0.420593927, 0.420593927, 0.420593927, 0.420593927, 0.420593927, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.4096636, 0.4096636, 0.402327216, 0.402327216, 0.402327216, 0.402327216, 0.402327216, 0.402327216, 0.402327216], [0.376502394, 0.376502394, 0.376502394, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.355207318, 0.355207318, 0.355207318, 0.355207318, 0.355207318, 0.355207318], [0.388383032, 0.388383032, 0.388383032, 0.376360702, 0.376360702, 0.376360702, 0.376360702, 0.376360702, 0.376360702, 0.372854205, 0.372854205, 0.372854205, 0.372854205, 0.372854205, 0.370352139, 0.370352139, 0.370352139, 0.370352139, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.353844203, 0.353844203, 0.353844203], [0.371717331, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839], [0.382143226, 0.382143226, 0.382143226, 0.382143226, 0.377237696, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.335860948, 0.335860948, 0.335860948, 0.335860948, 0.335860948], [0.369272184, 0.369272184, 0.357555506, 0.357555506, 0.357555506, 0.357555506, 0.357555506, 0.340464033, 0.340464033, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.333575233, 0.333575233, 0.333575233, 0.333575233, 0.333575233, 0.328385373, 0.328385373, 0.328385373, 0.325503262, 0.325503262, 0.325503262, 0.325503262]]
#voice_woa = [[0.407279847, 0.407279847, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.392265204, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592], [0.316470858, 0.316470858, 0.312178366, 0.312178366, 0.309226772, 0.309006432, 0.309006432, 0.309006432, 0.309006432, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308046134, 0.308046134, 0.308046134, 0.308046134, 0.306629648, 0.306629648, 0.306629648, 0.306629648], [0.21471561, 0.21471561, 0.21471561, 0.21471561, 0.21156003, 0.21156003, 0.21156003, 0.21156003, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322], [0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403], [0.18710837, 0.186330118, 0.186330118, 0.180373802, 0.180373802, 0.180373802, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.177026679, 0.177026679], [0.169878089, 0.169878089, 0.169878089, 0.16790808, 0.16790808, 0.16790808, 0.166822484, 0.166822484, 0.166822484, 0.166822484, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159054594, 0.159054594, 0.159054594], [0.154168241, 0.154168241, 0.15237128, 0.151141298, 0.151141298, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827]]


##print(len(list(zip(ga_sli, svd_woa, voice_woa))[0]))
#x1, x2, x3, x4, x5, x6, x7 = list(zip(*[ga_sli, svd_woa, voice_woa]))


x1 = ga_sli
x2 = svd_woa
x3 = voice_woa

#print(pd.DataFrame(ga_sli).T)

for i in range(1, 4):
  name = f'df{i}'
  data = f'x{i}'

  locals()[name] = pd.DataFrame(locals()[data]).T
  locals()[name].columns = ['0.2', '0.5', '0.8', '0.85', '0.9', '0.95', '0.999']


datasets = [df1, df2, df3]




A = 6
plt.rc('figure', figsize=[12, 8])
# Use Latex


# Define which colours you want to use
colours = ['blue', 'red', 'green']



# Set x-positions for boxes
x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
x_pos = (x_pos_range * 0.5) + 0.75
# Plot
for i, data in enumerate(datasets):
    bp = plt.boxplot(
        np.array(data), sym='', whis=[0, 100], widths=0.6 / len(datasets),
        labels=list(datasets[0]), patch_artist=True,
        positions=[x_pos[i] + j * 1 for j in range(len(data.T))]
    )

    medians = [item.get_ydata()[0] for item in bp['medians']]
    upper_quartile = [item.get_ydata()[0] for item in bp['whiskers'][:len(data)]]
    lower_quartile = [item.get_ydata()[1] for item in bp['whiskers'][:len(data)]]

    # Print the upper and lower quartiles  # Print the median values
    print("Median values:", medians)
    print("Upper quartile (Q3) values:", upper_quartile)
    print("Lower quartile (Q1) values:", lower_quartile)

    # Fill the boxes with colours (requires patch_artist=True)
    k = i % len(colours)
    for box in bp['boxes']:
        box.set(facecolor=colours[k])

    for element in ['boxes', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=colours[k])

    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=colours[k])
        plt.setp(bp[element], color=colours[k])


# Titles
#plt.title('Long Jump Finals at the Last Four Olympic Games')
plt.ylabel('F1-score', fontsize=13)
plt.xlabel('Alpha', fontsize=13)


# Axis ticks and labels
#plt.rc('figure', figsize=[15, 8])
plt.xticks(np.arange(len(list(datasets[0]))) + 1)
plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(
    np.array(range(len(list(datasets[0])) + 1)) + 0.5)
)
plt.gca().tick_params(axis='x', which='minor', length=4)
plt.gca().tick_params(axis='x', which='major', length=0)

legend_patches = [plt.Rectangle((0,0),1,1,fc=color, edgecolor='none') for color in colours]
box_labels = ['GA_SLI', 'WOA_SVD', 'WOA_VOICED']
plt.legend(legend_patches, box_labels, loc="upper right")
plt.grid(True)

# Change the limits of the x-axis
plt.xlim([0.5, len(list(datasets[0])) + 0.5])

plt.show()
plt.savefig('plot.pdf')



'''

import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib.pyplot as plt

ga_sli = [
    [0.398478437, 0.385006365, 0.385006365, 0.372570127, 0.372570127, 0.370928013, 0.367151133, 0.36593338, 0.364938344,
     0.363898409, 0.362320837, 0.362320837, 0.358686395, 0.358686395, 0.356736314, 0.356736314, 0.355132142,
     0.352930718, 0.350354333, 0.350354333, 0.350354333, 0.348172547, 0.347428547, 0.347428547, 0.345852653,
     0.345852653, 0.345576272, 0.34480807, 0.34480807, 0.340636979],
    [0.291130538, 0.276702735, 0.265066591, 0.254541529, 0.254541529, 0.251491051, 0.24840912, 0.24840912, 0.24840912,
     0.24840912, 0.24840912, 0.247504387, 0.247504387, 0.244822971, 0.244822971, 0.244822971, 0.244822971, 0.244822971,
     0.244822971, 0.244822971, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791, 0.238940791,
     0.238940791, 0.238940791, 0.238940791, 0.238940791],
    [0.159752494, 0.139631, 0.133888047, 0.132377724, 0.130425691, 0.127653684, 0.121674403, 0.121674403, 0.121557718,
     0.119818624, 0.119818624, 0.119035725, 0.119035725, 0.119035725, 0.119035725, 0.118456333, 0.11844567, 0.117088114,
     0.117088114, 0.117088114, 0.117088114, 0.117088114, 0.117088114, 0.116861448, 0.11684649, 0.11684649, 0.11684649,
     0.116778721, 0.116210705, 0.116162117],
    [0.123954867, 0.117087112, 0.117087112, 0.109586906, 0.109586906, 0.109586906, 0.102905123, 0.102905123,
     0.097963727, 0.097963727, 0.094630824, 0.094450182, 0.094450182, 0.094450182, 0.094450182, 0.09435127, 0.094308561,
     0.094308561, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534,
     0.091592534, 0.091592534, 0.091592534, 0.091592534, 0.091592534],
    [0.092587601, 0.089459824, 0.087585685, 0.085496557, 0.079120349, 0.079120349, 0.079120349, 0.079120349,
     0.079120349, 0.077652248, 0.077652248, 0.077652248, 0.077652248, 0.077652248, 0.074995365, 0.074995365,
     0.074995365, 0.074995365, 0.074995365, 0.074995365, 0.074328819, 0.074328819, 0.074328819, 0.073827859,
     0.073827859, 0.073827859, 0.073827859, 0.073827859, 0.073827859, 0.073827859],
    [0.090053852, 0.08575038, 0.079230846, 0.066765853, 0.056776105, 0.056776105, 0.056776105, 0.056145065, 0.056145065,
     0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.056145065, 0.055731699,
     0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055731699, 0.055000509,
     0.054552806, 0.054552806, 0.053157944, 0.052799949, 0.052799949],
    [0.073075102, 0.073075102, 0.05047963, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683, 0.050223683,
     0.050223683, 0.050223683, 0.050223683, 0.046021856, 0.046021856, 0.046021856, 0.046021856, 0.044582189,
     0.044582189, 0.044582189, 0.044582189, 0.034064057, 0.034064057, 0.032788502, 0.032788502, 0.032788502,
     0.032788502, 0.032788502, 0.028479114, 0.028479114, 0.026367306]]
svd_woa = [[0.488344873, 0.462057473, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098,
            0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098, 0.455927098,
            0.455927098, 0.455927098, 0.455927098, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394,
            0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394, 0.447803394],
           [0.434280413, 0.42963652, 0.429252614, 0.420593927, 0.420593927, 0.420593927, 0.420593927, 0.420593927,
            0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441,
            0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.413885441, 0.4096636, 0.4096636, 0.402327216,
            0.402327216, 0.402327216, 0.402327216, 0.402327216, 0.402327216, 0.402327216],
           [0.376502394, 0.376502394, 0.376502394, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524,
            0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524,
            0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524, 0.360007524,
            0.355207318, 0.355207318, 0.355207318, 0.355207318, 0.355207318, 0.355207318],
           [0.388383032, 0.388383032, 0.388383032, 0.376360702, 0.376360702, 0.376360702, 0.376360702, 0.376360702,
            0.376360702, 0.372854205, 0.372854205, 0.372854205, 0.372854205, 0.372854205, 0.370352139, 0.370352139,
            0.370352139, 0.370352139, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227, 0.356020227,
            0.356020227, 0.356020227, 0.356020227, 0.353844203, 0.353844203, 0.353844203],
           [0.371717331, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626, 0.357655626,
            0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.348841524, 0.333564839,
            0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839,
            0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839, 0.333564839],
           [0.382143226, 0.382143226, 0.382143226, 0.382143226, 0.377237696, 0.343173155, 0.343173155, 0.343173155,
            0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.343173155, 0.338443118, 0.338443118,
            0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118, 0.338443118,
            0.338443118, 0.335860948, 0.335860948, 0.335860948, 0.335860948, 0.335860948],
           [0.369272184, 0.369272184, 0.357555506, 0.357555506, 0.357555506, 0.357555506, 0.357555506, 0.340464033,
            0.340464033, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026, 0.338574026,
            0.338574026, 0.338574026, 0.333575233, 0.333575233, 0.333575233, 0.333575233, 0.333575233, 0.328385373,
            0.328385373, 0.328385373, 0.325503262, 0.325503262, 0.325503262, 0.325503262]]
voice_woa = [[0.407279847, 0.407279847, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277, 0.400668277,
              0.400668277, 0.400668277, 0.400668277, 0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.394642327,
              0.394642327, 0.394642327, 0.394642327, 0.394642327, 0.392265204, 0.383065592, 0.383065592, 0.383065592,
              0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592, 0.383065592],
             [0.316470858, 0.316470858, 0.312178366, 0.312178366, 0.309226772, 0.309006432, 0.309006432, 0.309006432,
              0.309006432, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453,
              0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308636453, 0.308046134, 0.308046134,
              0.308046134, 0.308046134, 0.306629648, 0.306629648, 0.306629648, 0.306629648],
             [0.21471561, 0.21471561, 0.21471561, 0.21471561, 0.21156003, 0.21156003, 0.21156003, 0.21156003,
              0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322,
              0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322,
              0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322, 0.210200322],
             [0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.201495989,
              0.201495989, 0.201495989, 0.201495989, 0.201495989, 0.198136028, 0.198136028, 0.198136028, 0.198136028,
              0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.198136028, 0.196855403, 0.196855403,
              0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403, 0.196855403],
             [0.18710837, 0.186330118, 0.186330118, 0.180373802, 0.180373802, 0.180373802, 0.179506018, 0.179506018,
              0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018,
              0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.179506018,
              0.179506018, 0.179506018, 0.179506018, 0.179506018, 0.177026679, 0.177026679],
             [0.169878089, 0.169878089, 0.169878089, 0.16790808, 0.16790808, 0.16790808, 0.166822484, 0.166822484,
              0.166822484, 0.166822484, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945,
              0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945, 0.159703945,
              0.159703945, 0.159703945, 0.159703945, 0.159054594, 0.159054594, 0.159054594],
             [0.154168241, 0.154168241, 0.15237128, 0.151141298, 0.151141298, 0.145034827, 0.145034827, 0.145034827,
              0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827,
              0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827,
              0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827, 0.145034827]]

##print(len(list(zip(ga_sli, svd_woa, voice_woa))[0]))
# x1, x2, x3, x4, x5, x6, x7 = list(zip(*[ga_sli, svd_woa, voice_woa]))


x1 = ga_sli
x2 = svd_woa
x3 = voice_woa

# print(pd.DataFrame(ga_sli).T)

for i in range(1, 4):
    name = f'df{i}'
    data = f'x{i}'

    locals()[name] = pd.DataFrame(locals()[data]).T
    locals()[name].columns = ['0.2', '0.5', '0.8', '0.85', '0.9', '0.95', '0.999']

datasets = [df1, df2, df3]

A = 6
plt.rc('figure', figsize=[12, 8])
# Use Latex


# Define which colours you want to use
colours = ['blue', 'red', 'green']

# Set x-positions for boxes
x_pos_range = np.arange(len(datasets)) / (len(datasets) - 1)
x_pos = (x_pos_range * 0.5) + 0.75
# Plot
for i, data in enumerate(datasets):
    bp = plt.boxplot(
        np.array(data), sym='', whis=[0, 100], widths=0.6 / len(datasets),
        labels=list(datasets[0]), patch_artist=True,
        positions=[x_pos[i] + j * 1 for j in range(len(data.T))]
    )

    # Fill the boxes with colours (requires patch_artist=True)
    k = i % len(colours)
    for box in bp['boxes']:
        box.set(facecolor=colours[k])

    for element in ['boxes', 'fliers', 'means', 'medians']:
        plt.setp(bp[element], color=colours[k])

    for element in ['whiskers', 'caps']:
        plt.setp(bp[element], color=colours[k])
        plt.setp(bp[element], color=colours[k])

# Titles
# plt.title('Long Jump Finals at the Last Four Olympic Games')
plt.ylabel('Fitness', fontsize=13)
plt.xlabel('Alpha', fontsize=13)

# Axis ticks and labels
# plt.rc('figure', figsize=[15, 8])
plt.xticks(np.arange(len(list(datasets[0]))) + 1)
plt.gca().xaxis.set_minor_locator(ticker.FixedLocator(
    np.array(range(len(list(datasets[0])) + 1)) + 0.5)
)
plt.gca().tick_params(axis='x', which='minor', length=4)
plt.gca().tick_params(axis='x', which='major', length=0)

legend_patches = [plt.Rectangle((0, 0), 1, 1, fc=color, edgecolor='none') for color in colours]
box_labels = ['GA_SLI', 'WOA_SVD', 'VOICED_WOA']
plt.legend(legend_patches, box_labels, loc="upper right")
plt.grid(True)

# Change the limits of the x-axis
plt.xlim([0.5, len(list(datasets[0])) + 0.5])

plt.show()
plt.savefig('plot.pdf')
'''