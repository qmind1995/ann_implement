FILE(REMOVE_RECURSE
  "libglui.pdb"
  "libglui.so"
  "libglui.so.2.37"
)

# Per-language clean rules from dependency scanning.
FOREACH(lang)
  INCLUDE(CMakeFiles/glui.dir/cmake_clean_${lang}.cmake OPTIONAL)
ENDFOREACH(lang)
