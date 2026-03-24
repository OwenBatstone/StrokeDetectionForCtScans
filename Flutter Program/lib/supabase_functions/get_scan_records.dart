//Function to get data for admin view

import 'package:supabase_flutter/supabase_flutter.dart';
import '../data_classes/paitient_summary.dart'; // ScanRecord, SliceRecord

class GetScanRecords {
  //supabase user
  final supabase = Supabase.instance.client;

  Future<List<PatientSummary>> fetchAllScansWithSlices() async {
    //get from scan_slices in response
    //gets per slice info
    final response = await supabase
        .from('Scan_Slices')
        .select('slices_Id, scan_id, file_url, overlay_file_url')
        .order('scan_id');

    //overview for by scan info
    final overviewRespone = await supabase
        .from('Scan_Overview')
        .select('scan_id, prediction, confidence, total_slices, run_by');

    //For Debugging
    // print('OverView Response: $overviewRespone');
    //final user = supabase.auth.currentUser;
    // print('Current user: ${user?.id}');
    // print('Session: ${supabase.auth.currentSession}');
    // print('Raw rows: $response');

    final rows =
        response
            as List<
              dynamic
            >; //dynamic list with rows retrived from scan_overview
    final overview = overviewRespone as List<dynamic>;

    //for Correct typing
    final Map<String, dynamic> overviewMap = {
      for (final o in overview) o['scan_id'] as String: o,
    };

    //group by scanId
    final Map<String, List<String>> sliceIdsByScan = {};
    final Map<String, List<String>> urlsByScan = {};
    final Map<String, List<String?>> overlayPathByScan = {};

    await Future.wait(
      rows.map((row) async {
        //map to nallable to allow for null masks and confidence
        final sliceId = row['slices_Id'] as String?;
        final scanId = row['scan_id'] as String?;
        final imagePath = row['file_url'] as String?;
        final overlayPath = row['overlay_file_url'] as String?;

        if (sliceId == null || scanId == null) return;

        //create url in object storage to retrive image for card
        String imageUrl = '';
        if (imagePath != null && imagePath.isNotEmpty) {
          imageUrl = await supabase.storage
              .from('scan_images')
              .createSignedUrl(imagePath, 3600);
        }

        //create signed url or retrive image of overlay for card
        String? overlayUrl = null;
        if (overlayPath != null && overlayPath.isNotEmpty) {
          overlayUrl = await supabase.storage
              .from('overlay_images')
              .createSignedUrl(overlayPath, 3600);
        }
        // can be empty
        sliceIdsByScan.putIfAbsent(scanId, () => []).add(sliceId);
        urlsByScan.putIfAbsent(scanId, () => []).add(imageUrl);
        overlayPathByScan.putIfAbsent(scanId, () => []).add(overlayUrl);
      }),
    );

    //map function to order data for use in cards
    return sliceIdsByScan.entries.map((e) {
      final scanId = e.key;
      final overview = overviewMap[scanId];

      return PatientSummary(
        scanId: scanId,
        run_by: overview?['run_by'] as String?,
        label: overview?['prediction'] as String? ?? 'Unknown',
        confidence: (overview?['confidence'] as num?)?.toDouble() ?? 0.0,
        perClassProb: [],
        slicesUsed: e.value.length,
        totalSlices: overview?['total_slices'] as int? ?? e.value.length,
        slicesIds: e.value,
        imageUrl: urlsByScan[scanId] ?? [],
        overlay_file_url: overlayPathByScan[scanId] ?? [],
      );
    }).toList()..sort((a, b) => a.scanId.compareTo(b.scanId));
  }
}
