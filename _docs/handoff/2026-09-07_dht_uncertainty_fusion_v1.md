# RLE sigma and DHT posterior uncertainty fusion verification

User explicitly authorized follow-up verification after point-line failure diagnosis. Source `data/pallet/results/dht_pose_integration_v1/live`, new output `live/uncertainty_fusion_v1`. PROTOCOL/PURPOSE written before actual extraction/calibration. Frozen original artifacts preserved. YOLO only; DOPE has no RLE head. Existing692images,3DHTseeds, realDEV52 reused; no new training or finaltest.

Assignments: dht_operator extract_yolo_uncertainty.py with preserved one2one sigmaheads and matching anchor/stride/letterbox, evalmodeonly; GPUfirst. side_edge_targets extract_dht_uncertainty.py+uncertainty_geometry.py sign-aligned originalunitline moments + anisotropicWLS, GPUsecond. Root calibration/selection/eval/driver/statisticalaudit. dht_visualization uncertainty_report.py + representativeQA.

Protocol: hash-order synthval256 split128scale-calibration128strength-selection, αpointxy/βlineheight-depth frommean squaredresidual/rawvariance, floor1px².3families point_sigma_only,line_uncertainty_only,point_line_uncertainty; λ[0,.0625,.25,1,4], displacementcapNone/.01diagonal; baselinefallback. Sameframe3seeds notindependent. Finish HTMLautoopen/actualwindowverify and Discordthrough existinghook; userauthorizationpersists.


## Completed evaluation

YOLO extraction692/692 baseline exact; DHT16608 argmax/line exact. Sparse voting peak numerical differences measured separately (max6.31e-4), earlier numerical-stop protocols preserved. Extra worst-peak frame three-repeat moment/variance check in NUMERICAL_STABILITY_DHT.json; no GT. No frozen source change.

Selected point_sigma_only λ.0625+cap.01diag, line_uncertainty_only λ.0625+no cap, point_line_uncertainty λ0. Real52 baseline9.4026/PCK.65144 → pointσ9.2783/.68670 → line-only8.9709/.70593; both-uncertainty retains baseline. Line-only Δmean-.4316 CI[-.6942,-.2334], PCK+5.4487pp CI[3.5256,7.4519]. Raw sigma error>10 AUROC.766 vs confidence.422, frame-bootstrap deltaCI[.2167,.4773]. Independent audit PASS11072 prediction rows,6228 independent weighted solutions,112 summary rows. GT-difficulty analysis stored in DIAGNOSIS.json and notes.

Cross_v4 line-only15.704→16.097 worse; reused realDEV and scene correlation limit inference. No further real-based tuning or new training warranted by current authorization/protocol.

Scripts: extract_yolo_uncertainty.py, extract_dht_uncertainty.py, uncertainty_geometry.py, evaluate_uncertainty.py, audit_uncertainty.py, diagnose_uncertainty.py, uncertainty_report.py, finalize_uncertainty.py. Root opened actual HTML through explicit Chrome helper and confirmed visible report window. Final visual QA+Discord delivery evidence in VISUAL_QA.json, GALLERY_OPEN.json, DISCORD_NOTIFICATION.json and COMPLETION.json.

- 최종 완료 영수증 확인: VISUAL_QA PASS, 실제 Chrome 창 표시 확인, Discord HTTP204 전송 성공. HTML SHA `3598246a1a09a9cafa0d9bdfdf1d86778a712cdcd1556d75ab74a464c7ba53e0`.
