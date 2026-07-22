# Prod vs Box vs Local Scratch — File Inventory

All comparisons are on canonical `.glb` filenames (`@YYYYMMDDTHHMMSS` upload
timestamps stripped on both sides).

Four stores compared:

- **prod** — from `mesh-original-filenames.txt` (endpoint output)
- **box source** — `.json` files under `Box/.../MIMC/Data/Domains/{Particles,Subunits}/`
- **box OLD backup** — `.glb` files under `Box/.../MIMC/Data/DomainMeshes/{Particles_OLD,Subunits_OLD}/`
- **local scratch** — `.glb` files under `~/local-scratch/{ParticleMeshes,SubunitMeshes}/`

---

# Particles

| store | count |
|---|---:|
| prod | 461 |
| box source (`.json`) | 475 |
| box OLD backup (`.glb`) | 471 |
| local scratch (`.glb`) | 475 |

## 1. Prod particles — do we have them elsewhere?

| filename | in box source | in box OLD backup | in local scratch |
|---|:-:|:-:|:-:|
| `Aleja_cubic_II001_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II002_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II003_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II_segment_150.glb` | ✓ |  | ✓ |
| `Kat_13_7_betatub555_1_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_betatub555_2_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_s100b555_6_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_s100b555_8_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_sox2555_1_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_sox2555_2_segment_300.glb` | ✓ | ✓ | ✓ |
| `Yining_L_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_L_2_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_L_3_segment_200.glb` | ✓ | ✓ | ✓ |
| `Yining_M_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_M_2_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_M_3_segment_200.glb` | ✓ | ✓ | ✓ |
| `Yining_S_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_S_2_segment_50.glb` | ✓ | ✓ | ✓ |
| `Yining_S_3_segment_100.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,00_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,02_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,04_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,05_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,06_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,08_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,10_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,12_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,15_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,20_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100,25_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100,100}_100_{0,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190,100}_{,100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90,100}_0_{100,0}_9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_anisotropic_ellipsoids.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_perfect_ellipsoids.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v10.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v0.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v6.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_50.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v0.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v6.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s02_{soft-0.25,hard-0.75}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s03_{soft-0.25,hard-0.75}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s04_{soft-0.25,hard-0.75}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s05_{soft-0.25,hard-0.75}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s06_{soft-0.50,hard-0.50}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s07_{soft-0.50,hard-0.50}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s08_{soft-0.50,hard-0.50}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s09_{soft-0.50,hard-0.50}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s10_{soft-0.50,hard-0.50}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s11_{soft-0.75,hard-0.25}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s12_{soft-0.75,hard-0.25}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s13_{soft-0.75,hard-0.25}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s14_{soft-0.75,hard-0.25}.glb` |  | ✓ |  |
| `labeledDomain_spheres_s15_{soft-0.75,hard-0.25}.glb` |  | ✓ |  |

**Summary:**
- Total prod particles: 461
- With a local scratch mesh: 446
- With a box source `.json`: 446
- With a box OLD backup: 457
- Missing from local scratch: 15
- Missing from box source: 15
- **Missing from all three: 0**

### Prod particles missing from local scratch (15):

- `labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s02_{soft-0.25,hard-0.75}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s03_{soft-0.25,hard-0.75}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s04_{soft-0.25,hard-0.75}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s05_{soft-0.25,hard-0.75}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s06_{soft-0.50,hard-0.50}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s07_{soft-0.50,hard-0.50}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s08_{soft-0.50,hard-0.50}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s09_{soft-0.50,hard-0.50}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s10_{soft-0.50,hard-0.50}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s11_{soft-0.75,hard-0.25}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s12_{soft-0.75,hard-0.25}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s13_{soft-0.75,hard-0.25}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s14_{soft-0.75,hard-0.25}.glb` (box source: —, box OLD: ✓)
- `labeledDomain_spheres_s15_{soft-0.75,hard-0.25}.glb` (box source: —, box OLD: ✓)

## 2. Local particles files NOT in prod (29)

- `beadInfo_spheres_square_040.glb`
- `beadInfo_stiffness_mixtures_a_{100,100}_100_{0,100}_0.glb`
- `beadInfo_stiffness_mixtures_a_{100,100}_100_{0,100}_1.glb`
- `beadInfo_stiffness_mixtures_a_{100,100}_100_{0,100}_2.glb`
- `beadInfo_stiffness_mixtures_a_{100,100}_100_{0,100}_3.glb`
- `beadInfo_stiffness_mixtures_a_{100,100}_100_{0,100}_4.glb`
- `inverse_spheres_hex_040.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s1.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s2.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s3.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s4.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s5.glb`
- `labeledDomain_nuggets_60_v5.glb`
- `labeledDomain_rods_80_v1.glb`
- `labeledDomain_spheres_s01.glb`
- `labeledDomain_spheres_s02.glb`
- `labeledDomain_spheres_s03.glb`
- `labeledDomain_spheres_s04.glb`
- `labeledDomain_spheres_s05.glb`
- `labeledDomain_spheres_s06.glb`
- `labeledDomain_spheres_s07.glb`
- `labeledDomain_spheres_s08.glb`
- `labeledDomain_spheres_s09.glb`
- `labeledDomain_spheres_s10.glb`
- `labeledDomain_spheres_s11.glb`
- `labeledDomain_spheres_s12.glb`
- `labeledDomain_spheres_s13.glb`
- `labeledDomain_spheres_s14.glb`
- `labeledDomain_spheres_s15.glb`

## 3. Box source particles `.json` files NOT in local (15)

- `labeledDomain_spheres_s01.{soft-0.25,hard-0.75}.glb`
- `labeledDomain_spheres_s02.{soft-0.25,hard-0.75}.glb`
- `labeledDomain_spheres_s03.{soft-0.25,hard-0.75}.glb`
- `labeledDomain_spheres_s04.{soft-0.25,hard-0.75}.glb`
- `labeledDomain_spheres_s05.{soft-0.25,hard-0.75}.glb`
- `labeledDomain_spheres_s06.{soft-0.50,hard-0.50}.glb`
- `labeledDomain_spheres_s07.{soft-0.50,hard-0.50}.glb`
- `labeledDomain_spheres_s08.{soft-0.50,hard-0.50}.glb`
- `labeledDomain_spheres_s09.{soft-0.50,hard-0.50}.glb`
- `labeledDomain_spheres_s10.{soft-0.50,hard-0.50}.glb`
- `labeledDomain_spheres_s11.{soft-0.75,hard-0.25}.glb`
- `labeledDomain_spheres_s12.{soft-0.75,hard-0.25}.glb`
- `labeledDomain_spheres_s13.{soft-0.75,hard-0.25}.glb`
- `labeledDomain_spheres_s14.{soft-0.75,hard-0.25}.glb`
- `labeledDomain_spheres_s15.{soft-0.75,hard-0.25}.glb`

---

# Pores

| store | count |
|---|---:|
| prod | 461 |
| box source (`.json`) | 473 |
| box OLD backup (`.glb`) | 469 |
| local scratch (`.glb`) | 473 |

## 1. Prod pores — do we have them elsewhere?

| filename | in box source | in box OLD backup | in local scratch |
|---|:-:|:-:|:-:|
| `Aleja_cubic_II001_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II002_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II003_segment_150.glb` | ✓ |  | ✓ |
| `Aleja_cubic_II_segment_150.glb` | ✓ |  | ✓ |
| `Kat_13_7_betatub555_1_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_betatub555_2_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_s100b555_6_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_s100b555_8_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_sox2555_1_segment_300.glb` | ✓ | ✓ | ✓ |
| `Kat_13_7_sox2555_2_segment_300.glb` | ✓ | ✓ | ✓ |
| `Yining_L_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_L_2_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_L_3_segment_200.glb` | ✓ | ✓ | ✓ |
| `Yining_M_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_M_2_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_M_3_segment_200.glb` | ✓ | ✓ | ✓ |
| `Yining_S_1_segment_100.glb` | ✓ | ✓ | ✓ |
| `Yining_S_2_segment_50.glb` | ✓ | ✓ | ✓ |
| `Yining_S_3_segment_100.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_mu70_sd15_v9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_00_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_02_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_04_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_05_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_06_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_08_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_10_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_12_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_15_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_20_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_01.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_02.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_03.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_04.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_05.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_06.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_07.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_08.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_09.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_normal_dist_100_25_10.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_uniform_40_100_v9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{100_100}_100_{0_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{110_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{120_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{130_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{140_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{150_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{160_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{170_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{180_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{190_100}_{_100}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{200}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{40_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{50_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{60_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{70_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{80_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_0.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_1.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_2.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_3.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_4.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_5.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_6.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_7.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_8.glb` | ✓ | ✓ | ✓ |
| `beadInfo_spheres_{90_100}_0_{100_0}_9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_anisotropic_ellipsoids.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_110_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_120_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_60_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_70_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_ellipsoids_90_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_70_5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_nuggets_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_perfect_ellipsoids.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rbd_isotropic_ellipsoids_s5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_110_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_120_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v10.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_60_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_70_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_80_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_rods_90_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v0.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v6.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_100_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_130_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_50.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_semi-soft-spheres_70_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v0.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v1.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v2.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v3.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v4.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v5.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v6.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v7.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v8.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_soft-spheres_100_v9.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s01_{soft-0_25_hard-0_75}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s02_{soft-0_25_hard-0_75}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s03_{soft-0_25_hard-0_75}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s04_{soft-0_25_hard-0_75}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s05_{soft-0_25_hard-0_75}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s06_{soft-0_50_hard-0_50}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s07_{soft-0_50_hard-0_50}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s08_{soft-0_50_hard-0_50}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s09_{soft-0_50_hard-0_50}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s10_{soft-0_50_hard-0_50}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s11_{soft-0_75_hard-0_25}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s12_{soft-0_75_hard-0_25}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s13_{soft-0_75_hard-0_25}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s14_{soft-0_75_hard-0_25}.glb` | ✓ | ✓ | ✓ |
| `labeledDomain_spheres_s15_{soft-0_75_hard-0_25}.glb` | ✓ | ✓ | ✓ |

**Summary:**
- Total prod pores: 461
- With a local scratch mesh: 461
- With a box source `.json`: 461
- With a box OLD backup: 457
- Missing from local scratch: 0
- Missing from box source: 0
- **Missing from all three: 0**

## 2. Local pores files NOT in prod (12)

- `beadInfo_spheres_square_040.glb`
- `beadInfo_stiffness_mixtures_a_{100_100}_100_{0_100}_0.glb`
- `beadInfo_stiffness_mixtures_a_{100_100}_100_{0_100}_1.glb`
- `beadInfo_stiffness_mixtures_a_{100_100}_100_{0_100}_2.glb`
- `beadInfo_stiffness_mixtures_a_{100_100}_100_{0_100}_3.glb`
- `beadInfo_stiffness_mixtures_a_{100_100}_100_{0_100}_4.glb`
- `inverse_spheres_hex_040.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s1.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s2.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s3.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s4.glb`
- `labeledDomain_FEM_Spheres_stiffness_mixtures_soft_s5.glb`

## 3. Box source pores `.json` files NOT in local (0)

_None._

---

# Notes on the counts that look weird

## Particles: 15 prod entries missing from local AND box source

All 15 are stiffness-mixture variants:
```
labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb
labeledDomain_spheres_s02_{soft-0.25,hard-0.75}.glb
... through s15 ...
```

They **do** exist as source `.json` in Box, but with a different name:
```
Box source: labeledDomain_spheres_s01.{soft-0.25,hard-0.75}.json   (dots)
Prod name:  labeledDomain_spheres_s01_{soft-0.25,hard-0.75}.glb    (underscore before {)
```

Because they don't match on canonical name, they show up as "missing."
Locally we DO have GLBs for them, but with the wrong names due to a bug in
`workflow_runner/mesh_generation.py:15,17` — it uses `.split('.')[0]` which
truncates at the first dot, yielding `labeledDomain_spheres_s01.glb`,
`s02.glb`, ... `s15.glb`. Those 15 filenames show up under "local not in
prod" for particles.

## Particles: local (475) > box source (475) = 1:1, but 29 local not in prod

Local and box source are 1:1. The 29 "local not in prod" are:
- 15 truncated names (`labeledDomain_spheres_sXX.glb`) — see above
- 14 legitimate extras that don't match a prod entry (never uploaded to prod)

## Particles: local scratch has 624 files but 475 unique names

149 canonical particle names appear twice locally — one copy in
`ParticleMeshes/Simulated/DatsJsonified/` and another in
`ParticleMeshes/Simulated/Jsons/`. That's because the Box source folder
`Box/Domains/Particles/Simulated/Jsons/` is a 149-file byte-identical
subset of `DatsJsonified/`. Same source JSONs, indexed twice → generated
twice → two output GLBs.

This contradicts the intent (`Jsons/` should hold JSONs distinct from the
jsonified DATs). Cleanup: purge the 149 duplicates from
`Box/Domains/Particles/Simulated/Jsons/` (or replace them with the JSONs it
was originally supposed to hold).

## Pores: 100% prod-covered, 12 local extras

461/461 prod pore names match a local subunit mesh. The 12 local extras
are subunit meshes we generated but that were never uploaded to prod.
Not a problem — just don't push them to prod.
