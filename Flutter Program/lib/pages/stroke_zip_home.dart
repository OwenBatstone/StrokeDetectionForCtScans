//Imports
import 'dart:io';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:stroketry3/pages/admin_view.dart';
import 'package:supabase_flutter/supabase_flutter.dart';
import 'package:uuid/uuid.dart';

//widgets
import '../widgets/slice_viewer_screen.dart';
import '../widgets/patient_summary_card.dart';

//import dataClasses

import '../data_classes/paitient_summary.dart';
import '../data_classes/slice_results.dart';

//import utils
import '../utils/ensure_png_bytes.dart';
import '../utils/softmax.dart';
import '../utils/argmax.dart';
import '../utils/decode_with_ui.dart';
import '../utils/extract_images_from_zip.dart';

//Supabase Functions
import '../supabase_functions/insert_patient_summary.dart';

//inference
import '../stroke_inference_service/stroke_inference_service.dart';
//admin check
import '../supabase_functions/is_user_admin.dart';

class StrokeZipHome extends StatefulWidget {
  const StrokeZipHome({super.key});
  @override
  State<StrokeZipHome> createState() => _StrokeZipHomeState();
}

class _StrokeZipHomeState extends State<StrokeZipHome> {
  final _svc = StrokeInferenceService();

  bool _busy = false;
  String _status = 'Upload a .zip with images (png/jpg/jpeg).';
  String _modelInfo = '';
  bool _isAdmin = true;

  PatientSummary? _summary;
  List<SliceResult> _rows = [];

  @override
  void initState() {
    super.initState();
    _loadAdminStatus();
  }

  @override
  void dispose() {
    _svc.dispose();
    super.dispose();
  }

  //Function to retrive admin status used for showing admin navigation
  Future<void> _loadAdminStatus() async {
    try {
      final adminStatus = await isAdmin();
      if (mounted) {
        setState(() {
          _isAdmin = adminStatus;
        });
      }
    } catch (e) {
      print('isAdmin error: $e');
      if (mounted) setState(() => _isAdmin = false);
    }
  }

  Future<void> _pickZipAndRun() async {
    setState(() {
      _busy = true; //disables button
      _status = 'Picking zip...'; //message indicating user is picking a zip
      _rows = []; //clears any results if they exist
      _summary = null; //same as above
    });

    try {
      final picked = await FilePicker.platform.pickFiles(
        type: FileType.custom,
        allowedExtensions: const [
          'zip',
        ], //makes the user have to pick a zip file **MAY WANT TO CHANGE**
        withData: true,
      );

      //if user cancels out of picking a files
      if (picked == null || picked.files.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'Cancelled.';
        });
        return;
      }

      final file = picked.files.single; //grabs the file
      final zipBytes =
          file.bytes ??
          await File(
            file.path!,
          ).readAsBytes(); //chcks for in memory bytes, if not reads from path

      //extracting file
      setState(() => _status = 'Extracting');
      final images = extractImagesFromZip(zipBytes);

      //error handeling, if theres no image inside the zip
      if (images.isEmpty) {
        setState(() {
          _busy = false;
          _status = 'No images found';
        });
        return;
      }

      setState(
        () => _status = 'Loading ONNX',
      ); //loading onnx file for classification and segmentation
      await _svc.ensureLoaded();
      setState(() => _modelInfo = _svc.modelInfo);

      setState(
        () => _status = 'Running inference on ${images.length} slices...',
      ); //shows how many slices are analyzed (takes about 1 mins per 100-200)

      //slice by slice results are here, give state at the end
      final out = <SliceResult>[];

      //sum of the slice averages logits
      final aggLogits = List<double>.filled(
        StrokeInferenceService.labels.length,
        0.0,
      );
      int usedForAgg = 0;

      //goes through each image in the zip
      for (final n in images) {
        final decoded = await decodeWithUi(
          n.bytes,
        ); //uses dartUI so we dont have to worry about weird formats

        if (decoded == null) {
          //if it failes record a new row and continue
          out.add(
            SliceResult(
              fileName: n.name, //orginial file name
              typeLabel: 'Decode failed', //failed, no guess
              confidence: 0, //no confidence (failed)
              logits: const [], //no Logits
              originalPng: n.bytes, //keep original bytes just in case
            ),
          );
          continue; //just skip this one
        }

        final pred = await _svc.predictType(decoded); //clasification on slice

        //if size matches, add logits to the patient-level accumulator
        if (pred.logits.isNotEmpty && pred.logits.length == aggLogits.length) {
          for (int i = 0; i < aggLogits.length; i++) {
            aggLogits[i] += pred.logits[i];
          }
          usedForAgg++; //counter for the amount of slices contributing
        }

        //Segmentation (to find Leissions)
        final seg = await _svc.predictMask(decoded);

        out.add(
          SliceResult(
            fileName: n.name,
            typeLabel: pred.label,
            confidence: pred.confidence,
            logits: pred.logits,

            ///raw logits
            originalPng: ensurePngBytes(
              decoded,
            ), //re-encoded to keep the display consistent
            maskOverlayPng: seg.overlayPng, //bytes of the overlay
            centroid: seg.centroid, //dot location
            maskScore: seg
                .maskScore, //about how much of the image is filled by the leision
          ),
        );
      }
      final supabase = Supabase.instance.client;
      //generate scan ID for batch
      final scanId = const Uuid().v4();
      //genereate sliceId for each slice in batch
      final sliceIds = List.generate(images.length, (_) => const Uuid().v4());

      final userId = supabase.auth.currentUser?.id;

      //uploading paitient summary
      setState(() => _status = 'Uploading slices'); //change text to uploading
      final imageUrls =
          <String>[]; // list for image Urls in supabase not nullable

      for (int i = 0; i < out.length; i++) {
        //for length of out
        final bytes = out[i].originalPng;
        if (bytes == null) {
          imageUrls.add('');
          continue;
        }
        final path = '$scanId/${sliceIds[i]}.png';
        await supabase.storage.from('scan_images').uploadBinary(path, bytes);
        imageUrls.add(path);
      }

      final overlayUrl = <String?>[];
      for (int i = 0; i < out.length; i++) {
        final bytes = out[i].maskOverlayPng;
        if (bytes == null) {
          overlayUrl.add(null);
          continue;
        }
        final path = '${scanId}/${sliceIds[i]}_overlay.png';
        await supabase.storage.from('overlay_images').uploadBinary(path, bytes);
        overlayUrl.add(path);
      }

      //Overall prediciton
      PatientSummary? summary; //checks if we have usable slices
      if (usedForAgg > 0) {
        //only compute if we have atleast one usable slice
        for (int i = 0; i < aggLogits.length; i++) {
          aggLogits[i] /= usedForAgg
              .toDouble(); //converts sumed logits to average logits
        }
        final probs = softmax(aggLogits); //turns average into probabilities
        final idx = argmax(probs); //find the label with the highest probability
        //Overall summary object *****COME HERE FOR DATABASE STUFF!!*****

        summary = PatientSummary(
          run_by: userId,
          label: StrokeInferenceService.labels[idx], //final choice
          confidence: probs[idx], //probability
          perClassProb: probs, //all classes probabilities
          slicesUsed: usedForAgg, //the slices that contributed
          totalSlices: images.length, //total slices in the zip
          scanId: scanId,
          slicesIds: sliceIds,
          imageUrl: imageUrls,
          overlay_file_url: overlayUrl,
        );
        print('usedForAgg: $usedForAgg');
        print('summary: $summary');
        print('calling insertPatientSummary...');
        await insertPatientSummary(summary);
        print('insert done');
      }
      //set the UI once at the end (way faster then doing it after each slice is ready, could change later if we value showing them as they come)
      setState(() {
        _busy = false;
        _rows = out;
        _summary = summary;
        _status = 'Done.';
      });
    } catch (e) {
      setState(() {
        _busy = false;
        _status = 'Error: $e';
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final summary = _summary;

    return Scaffold(
      //basic screen for now
      appBar: AppBar(title: const Text('Stroke ZIP Classifier + Locator')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            FilledButton.icon(
              onPressed: _busy ? null : _pickZipAndRun,
              icon: const Icon(Icons.upload_file),
              label: const Text('Upload ZIP + Run'),
            ),
            

            //button between pages
            if (_isAdmin)
            //box for spacing
            const SizedBox(height:10),
              //if admin add button under zip and navigate to admin on press
              FilledButton(
                onPressed: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => const AdminView()),
                ),
                child: Text("ADMIN PAGE"),
              ),
            const SizedBox(height: 10),
      
            Text(_status),
            if (_modelInfo.isNotEmpty) ...[
              const SizedBox(height: 10),
              Text(
                _modelInfo,
                style: const TextStyle(fontFamily: 'monospace', fontSize: 12),
              ),
            ],
            if (summary != null) ...[
              const SizedBox(height: 12),
              PatientSummaryCard(summary: summary),
            ],
            const SizedBox(height: 10),
            Expanded(
              child: _rows.isEmpty
                  ? const Center(child: Text('No results yet.'))
                  : ListView.separated(
                      itemCount: _rows.length,
                      separatorBuilder: (_, __) => const Divider(height: 1),
                      itemBuilder: (context, i) {
                        final r = _rows[i];
                        final hasMask =
                            r.maskOverlayPng != null && r.centroid != null;

                        return ListTile(
                          title: Text(r.fileName),
                          subtitle: Text(
                            'Type: ${r.typeLabel}  |  ${(r.confidence * 100).toStringAsFixed(1)}%'
                            '${hasMask ? '  |  Mask: ${(r.maskScore * 100).toStringAsFixed(1)}%' : ''}',
                          ),
                          trailing: hasMask
                              ? const Icon(Icons.image_search)
                              : null,
                          onTap: (r.originalPng == null)
                              ? null
                              : () {
                                  Navigator.of(context).push(
                                    MaterialPageRoute(
                                      builder: (_) =>
                                          SliceViewerScreen(result: r),
                                    ),
                                  );
                                },
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
