equation = "0.456 f0(L_sav1) + 0.182 f0(L_sav2) + 0.177 f0(L_sav3) + -0.141 f0(L_sav5) + -0.380 f0(L_sav6) + -0.327 f0(Ld1_3) + 0.163 f0(Ld1_4) + -0.160 f0(Ld1_5) + 0.258 f0(Ld1_6) + -0.549 f1(L_sav1,L_sav2) + -3.507 f1(L_sav1,L_sav3) + 1.284 f1(L_sav1,L_sav5) + -0.146 f1(L_sav1,L_sav6) + 0.217 f1(L_sav1,Ld1_1) + 0.742 f1(L_sav1,Ld1_2) + 2.041 f1(L_sav1,Ld1_3) + -0.849 f1(L_sav1,Ld1_4) + 0.185 f1(L_sav1,Ld1_5) + -0.860 f1(L_sav1,Ld1_6) + -0.095 f1(L_sav2,L_sav4) + 0.161 f1(L_sav2,L_sav6) + 0.251 f1(L_sav2,Ld1_3) + 0.386 f1(L_sav2,Ld1_4) + -0.115 f1(L_sav2,Ld1_5) + 1.042 f1(L_sav3,L_sav4) + -0.053 f1(L_sav3,L_sav5) + -1.331 f1(L_sav3,L_sav6) + 0.185 f1(L_sav3,Ld1_1) + -0.139 f1(L_sav3,Ld1_3) + -0.925 f1(L_sav3,Ld1_4) + 0.325 f1(L_sav3,Ld1_6) + -0.015 f1(L_sav4,L_sav5) + -0.806 f1(L_sav4,L_sav6) + -0.135 f1(L_sav4,Ld1_2) + -0.643 f1(L_sav4,Ld1_3) + -0.797 f1(L_sav4,Ld1_4) + 0.050 f1(L_sav4,Ld1_5) + -1.487 f1(L_sav4,Ld1_6) + -0.040 f1(L_sav4,F0) + 0.086 f1(L_sav5,L_sav6) + -0.105 f1(L_sav5,Ld1_2) + -0.132 f1(L_sav5,Ld1_4) + 0.086 f1(L_sav5,Ld1_6) + -0.009 f1(L_sav6,Ld1_1) + -0.291 f1(L_sav6,Ld1_3) + 0.227 f1(L_sav6,Ld1_4) + 0.165 f1(L_sav6,Ld1_5) + 0.122 f1(Ld1_1,Ld1_3) + 0.073 f1(Ld1_3,Ld1_4) + -0.045 f2(L_sav2) + 1.095 f2(L_sav3) + 1.177 f2(L_sav4) + 0.100 f2(L_sav5) + -0.470 f2(L_sav6) + 0.068 f2(Ld1_2) + 0.351 f2(Ld1_3) + -0.067 f2(Ld1_4) + 0.026 f2(Ld1_6) + -3.504 f3(L_sav1) + -0.192 f3(L_sav2) + 1.197 f3(L_sav3) + -0.264 f3(L_sav4) + 0.289 f3(L_sav5) + -0.797 f3(L_sav6) + -0.175 f3(Ld1_3) + -0.222 f3(Ld1_4) + 0.176 f3(Ld1_6) + -0.743 f4(L_sav1) + 0.669 f4(L_sav3) + -0.187 f4(L_sav6) + 0.160 f4(Ld1_4) + 0.440 f4(Ld1_6) " 



i_impl=0;
i_explSplit3=0;
i_explSplit2=0;
i_explSplit1=0;
i_VAE=0;
i_VAEDiff2=1;

%(Ld1_1)' = -0.910 f0(L_sav1) + -0.069 f0(L_sav4) + -0.247 f0(L_sav5) + 0.101 f0(Ld1_1) + -0.119 f0(Ld1_2) + -0.100 f0(Ld1_3) + -0.109 f0(Ld1_4) + 0.216 f0(Ld1_6) + 0.967 f1(L_sav1,L_sav3) + 0.218 f1(L_sav1,L_sav4) + 0.165 f1(L_sav1,Ld1_1) + 0.394 f1(L_sav1,Ld1_4) + 0.429 f1(L_sav1,Ld1_6) + -0.112 f1(L_sav2,L_sav3) + 0.474 f1(L_sav2,L_sav5) + -0.208 f1(L_sav2,Ld1_1) + 0.225 f1(L_sav2,Ld1_2) + 0.359 f1(L_sav2,Ld1_3) + -0.181 f1(L_sav2,Ld1_6) + 0.698 f1(L_sav3,L_sav5) + -0.331 f1(L_sav3,L_sav6) + -0.390 f1(L_sav3,Ld1_1) + -0.330 f1(L_sav3,Ld1_2) + -0.164 f1(L_sav3,Ld1_4) + -0.416 f1(L_sav3,Ld1_6) + -0.229 f1(L_sav4,L_sav5) + 0.113 f1(L_sav4,Ld1_1) + 0.191 f1(L_sav4,Ld1_2) + -0.322 f1(L_sav4,Ld1_3) + 0.139 f1(L_sav4,Ld1_4) + 0.365 f1(L_sav4,Ld1_5) + -0.133 f1(L_sav5,Ld1_2) + -0.294 f1(L_sav5,Ld1_3) + -0.130 f1(L_sav5,Ld1_4) + 0.214 f1(L_sav5,Ld1_5) + 0.143 f1(L_sav6,Ld1_1) + 0.283 f1(L_sav6,Ld1_4) + 0.077 f1(Ld1_1,Ld1_4) + 0.045 f1(Ld1_2,Ld1_3) + -0.394 f2(L_sav1) + 1.014 f2(L_sav3) + -0.432 f2(L_sav4) + 0.168 f2(L_sav5) + -0.084 f2(Ld1_1) + 0.082 f2(Ld1_2) + 0.337 f2(Ld1_6) + -0.187 f3(L_sav1) + 0.412 f3(L_sav3) + 4.032 f3(L_sav4) + 0.285 f3(L_sav6) + 0.059 f3(Ld1_1) + 0.546 f3(Ld1_3) + 0.110 f3(Ld1_4) + -0.264 f3(Ld1_5) + 0.105 f3(Ld1_6) + 0.297 f4(L_sav1) + 0.286 f4(L_sav3) + 0.130 f4(L_sav4) + 0.178 f4(Ld1_1) + -0.120 f4(Ld1_3) + -0.071 f4(Ld1_4) + -0.051 f4(Ld1_6) 
%(Ld1_2)' = -0.027 f0(L_sav1) + -0.908 f0(L_sav2) + 1.914 f0(L_sav3) + 0.362 f0(L_sav4) + 0.381 f0(L_sav5) + -0.376 f0(L_sav6) + 0.143 f0(Ld1_1) + 0.290 f0(Ld1_2) + 0.480 f0(Ld1_3) + -0.461 f0(Ld1_4) + -0.267 f0(Ld1_5) + 0.436 f0(Ld1_6) + 0.828 f1(L_sav1,L_sav2) + -2.985 f1(L_sav1,L_sav3) + 0.326 f1(L_sav1,L_sav4) + 0.409 f1(L_sav1,L_sav5) + -0.188 f1(L_sav1,L_sav6) + -0.620 f1(L_sav1,Ld1_1) + 0.326 f1(L_sav1,Ld1_2) + 1.851 f1(L_sav1,Ld1_3) + -0.321 f1(L_sav1,Ld1_5) + -2.557 f1(L_sav1,Ld1_6) + 0.069 f1(L_sav2,L_sav3) + 0.117 f1(L_sav2,L_sav4) + -1.029 f1(L_sav2,L_sav5) + 0.174 f1(L_sav2,Ld1_1) + -0.792 f1(L_sav2,Ld1_2) + -1.518 f1(L_sav2,Ld1_3) + -0.531 f1(L_sav2,Ld1_4) + 0.332 f1(L_sav2,Ld1_5) + 0.295 f1(L_sav2,Ld1_6) + 1.125 f1(L_sav3,L_sav4) + -0.801 f1(L_sav3,L_sav5) + 1.060 f1(L_sav3,L_sav6) + 0.586 f1(L_sav3,Ld1_1) + 0.894 f1(L_sav3,Ld1_2) + -0.415 f1(L_sav3,Ld1_3) + -0.111 f1(L_sav3,Ld1_4) + 0.611 f1(L_sav3,Ld1_5) + -0.507 f1(L_sav3,Ld1_6) + -0.663 f1(L_sav4,L_sav5) + 0.107 f1(L_sav4,L_sav6) + 2.446 f1(L_sav4,Ld1_1) + 0.692 f1(L_sav4,Ld1_2) + -0.894 f1(L_sav4,Ld1_3) + -0.074 f1(L_sav4,Ld1_5) + 2.524 f1(L_sav4,Ld1_6) + 0.094 f1(L_sav4,F0) + 0.003 f1(L_sav5,L_sav6) + 0.152 f1(L_sav5,Ld1_2) + 0.846 f1(L_sav5,Ld1_3) + 0.572 f1(L_sav5,Ld1_4) + -0.550 f1(L_sav5,Ld1_5) + -0.521 f1(L_sav5,Ld1_6) + -0.343 f1(L_sav6,Ld1_1) + -0.285 f1(L_sav6,Ld1_2) + 0.171 f1(L_sav6,Ld1_3) + 0.144 f1(L_sav6,Ld1_4) + 0.022 f1(L_sav6,Ld1_5) + -0.536 f1(L_sav6,Ld1_6) + -0.150 f1(Ld1_1,Ld1_2) + -0.361 f1(Ld1_1,Ld1_3) + -0.176 f1(Ld1_1,Ld1_4) + 0.129 f1(Ld1_1,Ld1_5) + -0.264 f1(Ld1_2,Ld1_3) + -0.054 f1(Ld1_2,Ld1_4) + 0.045 f1(Ld1_2,Ld1_5) + 0.088 f1(Ld1_3,Ld1_5) + -0.090 f1(Ld1_3,Ld1_6) + 0.183 f1(Ld1_5,Ld1_6) + -1.960 f2(L_sav1) + -0.625 f2(L_sav2) + -1.588 f2(L_sav3) + 6.163 f2(L_sav4) + -0.393 f2(L_sav5) + 1.736 f2(L_sav6) + 0.190 f2(Ld1_1) + -0.179 f2(Ld1_2) + -0.249 f2(Ld1_3) + -0.581 f2(Ld1_6) + 5.279 f3(L_sav1) + -0.808 f3(L_sav2) + 2.829 f3(L_sav3) + 1.894 f3(L_sav4) + 0.402 f3(L_sav5) + 0.466 f3(Ld1_2) + -1.303 f3(Ld1_3) + -0.325 f3(Ld1_4) + 0.798 f3(Ld1_5) + -0.481 f4(L_sav1) + 0.040 f4(L_sav2) + -1.497 f4(L_sav3) + -0.068 f4(L_sav4) + 0.058 f4(L_sav5) + -0.298 f4(Ld1_1) + -0.099 f4(Ld1_2) + 0.318 f4(Ld1_3) + 0.145 f4(Ld1_4) + -0.118 f4(Ld1_5) + -0.021 f4(Ld1_6) 
%(Ld1_3)' = 0.249 f0(L_sav1) + 0.079 f0(L_sav2) + -0.882 f0(L_sav3) + 0.062 f0(L_sav6) + -0.239 f0(Ld1_1) + -0.184 f0(Ld1_3) + -0.310 f1(L_sav1,L_sav2) + 0.416 f1(L_sav1,L_sav3) + 0.230 f1(L_sav1,L_sav5) + 0.106 f1(L_sav1,L_sav6) + 0.253 f1(L_sav1,Ld1_2) + 0.196 f1(L_sav1,Ld1_3) + 0.363 f1(L_sav1,Ld1_6) + 0.048 f1(L_sav2,L_sav4) + 0.262 f1(L_sav2,Ld1_3) + 0.325 f1(L_sav2,Ld1_4) + -0.187 f1(L_sav2,Ld1_5) + -0.327 f1(L_sav2,Ld1_6) + -0.535 f1(L_sav3,L_sav6) + -0.172 f1(L_sav3,Ld1_2) + 1.083 f1(L_sav3,Ld1_3) + -0.263 f1(L_sav3,Ld1_5) + 0.140 f1(L_sav4,L_sav5) + -0.778 f1(L_sav4,L_sav6) + 0.420 f1(L_sav4,Ld1_2) + -0.537 f1(L_sav4,Ld1_4) + -0.390 f1(L_sav4,Ld1_6) + 0.047 f1(L_sav5,Ld1_2) + -0.207 f1(L_sav6,Ld1_1) + -0.128 f1(L_sav6,Ld1_2) + -0.232 f1(L_sav6,Ld1_3) + 0.152 f1(Ld1_2,Ld1_3) + 0.125 f1(Ld1_3,Ld1_6) + -0.070 f1(Ld1_5,Ld1_6) + 0.762 f2(L_sav1) + 0.199 f2(L_sav2) + -0.185 f2(L_sav5) + -0.695 f2(L_sav6) + 0.081 f2(Ld1_2) + 0.229 f2(Ld1_6) + 0.065 f3(L_sav5) + -0.303 f3(L_sav6) + 0.290 f3(Ld1_3) + 0.334 f3(Ld1_4) + 0.016 f3(Ld1_6) + 0.271 f4(L_sav3) + 0.146 f4(L_sav4) + 0.138 f4(Ld1_1) + 0.074 f4(Ld1_6) 
%(Ld1_4)' = 0.238 f0(L_sav1) + 0.234 f0(L_sav3) + -0.692 f0(L_sav4) + 0.162 f0(L_sav6) + 0.340 f0(Ld1_1) + 0.048 f0(Ld1_2) + -0.035 f0(Ld1_4) + 0.346 f0(Ld1_6) + 0.106 f1(L_sav1,L_sav2) + -0.711 f1(L_sav1,L_sav3) + 0.377 f1(L_sav1,L_sav4) + 0.175 f1(L_sav1,L_sav5) + 0.492 f1(L_sav1,Ld1_2) + -0.465 f1(L_sav1,Ld1_4) + -0.240 f1(L_sav1,Ld1_5) + -0.849 f1(L_sav1,Ld1_6) + 0.175 f1(L_sav2,L_sav3) + -0.289 f1(L_sav2,L_sav5) + 0.106 f1(L_sav2,Ld1_1) + -0.264 f1(L_sav2,Ld1_3) + 0.127 f1(L_sav2,Ld1_5) + 0.087 f1(L_sav2,Ld1_6) + 0.512 f1(L_sav3,L_sav4) + -0.066 f1(L_sav3,L_sav6) + -0.141 f1(L_sav3,Ld1_1) + 0.384 f1(L_sav3,Ld1_2) + -0.195 f1(L_sav3,Ld1_3) + -0.398 f1(L_sav3,Ld1_4) + -0.463 f1(L_sav3,Ld1_6) + 0.149 f1(L_sav4,L_sav5) + -0.468 f1(L_sav4,L_sav6) + 0.739 f1(L_sav4,Ld1_1) + 0.412 f1(L_sav4,Ld1_2) + -0.571 f1(L_sav4,Ld1_3) + -0.762 f1(L_sav4,Ld1_4) + 0.175 f1(L_sav4,Ld1_5) + -0.074 f1(L_sav5,Ld1_1) + -0.120 f1(L_sav5,Ld1_6) + 0.511 f1(L_sav6,Ld1_1) + 0.201 f1(L_sav6,Ld1_5) + -0.223 f1(L_sav6,Ld1_6) + 0.146 f1(Ld1_1,Ld1_3) + 0.125 f1(Ld1_1,Ld1_4) + -0.081 f1(Ld1_2,Ld1_6) + 0.116 f1(Ld1_5,Ld1_6) + -0.214 f2(L_sav2) + -0.245 f2(L_sav3) + 0.550 f2(L_sav4) + -0.136 f2(L_sav5) + 0.505 f2(L_sav6) + 0.182 f2(Ld1_6) + 0.018 f3(L_sav1) + 0.751 f3(L_sav3) + 0.851 f3(L_sav4) + 0.287 f3(L_sav6) + 0.307 f3(Ld1_1) + 0.121 f3(Ld1_2) + -0.247 f3(Ld1_4) + 0.269 f3(Ld1_5) + 0.246 f3(Ld1_6) + -0.145 f4(L_sav3) + -0.152 f4(Ld1_1) + -0.118 f4(Ld1_3) + 0.175 f4(Ld1_4) 
%(Ld1_5)' = -0.644 f0(L_sav1) + 0.095 f0(L_sav2) + 1.338 f0(L_sav3) + -0.006 f0(L_sav4) + -0.210 f0(L_sav5) + -0.185 f0(Ld1_1) + 0.334 f0(Ld1_2) + 0.385 f0(Ld1_3) + -0.133 f0(Ld1_4) + -0.183 f0(Ld1_5) + 0.276 f0(Ld1_6) + -4.571 f1(L_sav1,L_sav3) + -1.668 f1(L_sav1,L_sav4) + 0.676 f1(L_sav1,L_sav6) + -0.339 f1(L_sav1,Ld1_1) + -0.389 f1(L_sav1,Ld1_3) + -1.324 f1(L_sav1,Ld1_4) + 0.775 f1(L_sav1,Ld1_5) + 0.258 f1(L_sav1,Ld1_6) + -0.100 f1(L_sav1,F0) + -0.592 f1(L_sav2,L_sav3) + 0.591 f1(L_sav2,L_sav4) + -0.150 f1(L_sav2,L_sav5) + 0.026 f1(L_sav2,L_sav6) + 0.784 f1(L_sav2,Ld1_1) + 0.385 f1(L_sav2,Ld1_2) + -0.155 f1(L_sav2,Ld1_3) + -0.131 f1(L_sav2,Ld1_4) + 2.588 f1(L_sav3,L_sav4) + -1.578 f1(L_sav3,L_sav5) + -1.079 f1(L_sav3,L_sav6) + 0.275 f1(L_sav3,Ld1_1) + 1.477 f1(L_sav3,Ld1_2) + 0.978 f1(L_sav3,Ld1_3) + -0.915 f1(L_sav3,Ld1_5) + -0.875 f1(L_sav3,Ld1_6) + 0.461 f1(L_sav4,L_sav5) + -1.760 f1(L_sav4,L_sav6) + -0.643 f1(L_sav4,Ld1_1) + 0.450 f1(L_sav4,Ld1_2) + 0.399 f1(L_sav4,Ld1_3) + -1.021 f1(L_sav4,Ld1_4) + -1.869 f1(L_sav4,Ld1_6) + 0.254 f1(L_sav5,Ld1_1) + 0.062 f1(L_sav5,Ld1_2) + 0.129 f1(L_sav5,Ld1_3) + 0.504 f1(L_sav5,Ld1_4) + -0.143 f1(L_sav5,Ld1_5) + -0.271 f1(L_sav5,Ld1_6) + 0.410 f1(L_sav6,Ld1_1) + -0.466 f1(L_sav6,Ld1_3) + -0.782 f1(L_sav6,Ld1_4) + 0.188 f1(L_sav6,Ld1_5) + 0.079 f1(L_sav6,F0) + 0.124 f1(Ld1_1,Ld1_3) + -0.109 f1(Ld1_1,Ld1_4) + 0.148 f1(Ld1_2,Ld1_4) + -0.394 f1(Ld1_2,Ld1_6) + -0.029 f1(Ld1_3,Ld1_4) + 0.169 f1(Ld1_3,Ld1_6) + -0.033 f1(Ld1_4,Ld1_6) + 0.085 f1(Ld1_5,Ld1_6) + 5.852 f2(L_sav1) + -0.017 f2(L_sav2) + -1.178 f2(L_sav3) + 1.160 f2(L_sav4) + -1.261 f2(L_sav5) + -0.262 f2(L_sav6) + -0.079 f2(Ld1_2) + 0.296 f2(Ld1_3) + 0.004 f2(Ld1_6) + -5.128 f3(L_sav1) + -1.144 f3(L_sav2) + 6.483 f3(L_sav3) + -0.934 f3(L_sav4) + -1.112 f3(L_sav5) + 1.432 f3(L_sav6) + -0.315 f3(Ld1_1) + 0.345 f3(Ld1_2) + -0.521 f3(Ld1_3) + -0.528 f3(Ld1_4) + 0.401 f3(Ld1_5) + -0.285 f3(Ld1_6) + 0.028 f4(L_sav1) + -0.620 f4(L_sav3) + 0.188 f4(L_sav4) + -0.136 f4(L_sav5) + -0.002 f4(L_sav6) + 0.291 f4(Ld1_4) + 0.062 f4(Ld1_5) 
%(Ld1_6)' = 0.456 f0(L_sav1) + 0.182 f0(L_sav2) + 0.177 f0(L_sav3) + -0.141 f0(L_sav5) + -0.380 f0(L_sav6) + -0.327 f0(Ld1_3) + 0.163 f0(Ld1_4) + -0.160 f0(Ld1_5) + 0.258 f0(Ld1_6) + -0.549 f1(L_sav1,L_sav2) + -3.507 f1(L_sav1,L_sav3) + 1.284 f1(L_sav1,L_sav5) + -0.146 f1(L_sav1,L_sav6) + 0.217 f1(L_sav1,Ld1_1) + 0.742 f1(L_sav1,Ld1_2) + 2.041 f1(L_sav1,Ld1_3) + -0.849 f1(L_sav1,Ld1_4) + 0.185 f1(L_sav1,Ld1_5) + -0.860 f1(L_sav1,Ld1_6) + -0.095 f1(L_sav2,L_sav4) + 0.161 f1(L_sav2,L_sav6) + 0.251 f1(L_sav2,Ld1_3) + 0.386 f1(L_sav2,Ld1_4) + -0.115 f1(L_sav2,Ld1_5) + 1.042 f1(L_sav3,L_sav4) + -0.053 f1(L_sav3,L_sav5) + -1.331 f1(L_sav3,L_sav6) + 0.185 f1(L_sav3,Ld1_1) + -0.139 f1(L_sav3,Ld1_3) + -0.925 f1(L_sav3,Ld1_4) + 0.325 f1(L_sav3,Ld1_6) + -0.015 f1(L_sav4,L_sav5) + -0.806 f1(L_sav4,L_sav6) + -0.135 f1(L_sav4,Ld1_2) + -0.643 f1(L_sav4,Ld1_3) + -0.797 f1(L_sav4,Ld1_4) + 0.050 f1(L_sav4,Ld1_5) + -1.487 f1(L_sav4,Ld1_6) + -0.040 f1(L_sav4,F0) + 0.086 f1(L_sav5,L_sav6) + -0.105 f1(L_sav5,Ld1_2) + -0.132 f1(L_sav5,Ld1_4) + 0.086 f1(L_sav5,Ld1_6) + -0.009 f1(L_sav6,Ld1_1) + -0.291 f1(L_sav6,Ld1_3) + 0.227 f1(L_sav6,Ld1_4) + 0.165 f1(L_sav6,Ld1_5) + 0.122 f1(Ld1_1,Ld1_3) + 0.073 f1(Ld1_3,Ld1_4) + -0.045 f2(L_sav2) + 1.095 f2(L_sav3) + 1.177 f2(L_sav4) + 0.100 f2(L_sav5) + -0.470 f2(L_sav6) + 0.068 f2(Ld1_2) + 0.351 f2(Ld1_3) + -0.067 f2(Ld1_4) + 0.026 f2(Ld1_6) + -3.504 f3(L_sav1) + -0.192 f3(L_sav2) + 1.197 f3(L_sav3) + -0.264 f3(L_sav4) + 0.289 f3(L_sav5) + -0.797 f3(L_sav6) + -0.175 f3(Ld1_3) + -0.222 f3(Ld1_4) + 0.176 f3(Ld1_6) + -0.743 f4(L_sav1) + 0.669 f4(L_sav3) + -0.187 f4(L_sav6) + 0.160 f4(Ld1_4) + 0.440 f4(Ld1_6) 




modifiedEquation = regexprep(equation, "(Q|f|F|L|u|sin|cos)", "*$1");
modifiedEquation = regexprep(modifiedEquation, '\(\*', '(');
modifiedEquation = regexprep(modifiedEquation, '\,\*', ',');
modifiedEquation = regexprep(modifiedEquation, '+ -', '-');
if i_impl==1
    modifiedEquation = regexprep(modifiedEquation, 'Q_dot', 'yp(1)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_dot', 'yp(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_dot', 'yp(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_dot', 'yp(4)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1', 'yp(1)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2', 'yp(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3', 'yp(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Q', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'x0_dot', 'yp(1)');
    modifiedEquation = regexprep(modifiedEquation, 'x1_dot', 'yp(2)');
    modifiedEquation = regexprep(modifiedEquation, 'x2_dot', 'yp(3)');
    modifiedEquation = regexprep(modifiedEquation, 'x3_dot', 'yp(4)');
    modifiedEquation = regexprep(modifiedEquation, 'x1', 'yp(1)');
    modifiedEquation = regexprep(modifiedEquation, 'x2', 'yp(2)');
    modifiedEquation = regexprep(modifiedEquation, 'x3', 'yp(3)');
    modifiedEquation = regexprep(modifiedEquation, 'x0', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'u0', 'F0');
    disp(modifiedEquation);
end
if i_explSplit3==1
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav1', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav2', 'y(5)');
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav3', 'y(9)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_1', 'y(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_2', 'y(6)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_3', 'y(10)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_1', 'y(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_2', 'y(7)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_3', 'y(11)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_1', 'y(4)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_2', 'y(8)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_3', 'y(12)');
    disp(modifiedEquation);
end
if i_explSplit2==1
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav1', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav2', 'y(5)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_1', 'y(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_2', 'y(6)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_1', 'y(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_2', 'y(7)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_1', 'y(4)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_2', 'y(8)');
    disp(modifiedEquation);
end
if i_explSplit1==1
    modifiedEquation = regexprep(modifiedEquation, 'Q_sav1', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd2_1', 'y(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd3_1', 'y(4)');
    modifiedEquation = regexprep(modifiedEquation, 'Qd1_1', 'y(2)');
    disp(modifiedEquation);
end
if i_VAE==1
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_1', 'y(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_2', 'y(5)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_3', 'y(8)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_4', 'y(11)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_5', 'y(14)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_6', 'y(17)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_1', 'y(3)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_2', 'y(6)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_3', 'y(9)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_4', 'y(12)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_5', 'y(15)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld2_6', 'y(18)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav1', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav2', 'y(4)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav3', 'y(7)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav4', 'y(10)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav5', 'y(13)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav6', 'y(16)');
    disp(modifiedEquation);
end
if i_VAEDiff2==1
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_1', 'y(2)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_2', 'y(4)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_3', 'y(6)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_4', 'y(8)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_5', 'y(10)');
    modifiedEquation = regexprep(modifiedEquation, 'Ld1_6', 'y(12)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav1', 'y(1)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav2', 'y(3)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav3', 'y(5)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav4', 'y(7)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav5', 'y(9)');
    modifiedEquation = regexprep(modifiedEquation, 'L_sav6', 'y(11)');
    disp(modifiedEquation);
end
