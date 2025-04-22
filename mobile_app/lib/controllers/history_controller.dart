import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:cloudinary/cloudinary.dart';
import 'package:get/get.dart';
import 'package:mobile_app/P.dart';
import 'package:mobile_app/models/History.dart';

class HistoryController extends GetxController {
  Stream<List<Map<String, dynamic>>> getUserHistory() {
    return P.fireStore
        .collection("History")
        .where("uidUser", isEqualTo: P.auth.currentUser.value?.uid)
        .snapshots()
        .map((e) {
          return e.docs.map((doc) {
            return doc.data();
          }).toList();
        });
  }

  Future<void> clearHistory() async {
    try {
      var querySnapshot =
          await P.fireStore
              .collection("History")
              .where("uidUser", isEqualTo: P.auth.currentUser.value?.uid)
              .get();

      for (var doc in querySnapshot.docs) {
        await doc.reference.delete();
        String publicId = doc.data()['publicId'];
        await deleteImg(publicId);
      }
    } catch (e) {
      print("Lỗi delete: ${e.toString()}");
    }
  }
  Future<void> deleteImg(String publicId) async {
    P.cloudinary.destroy(publicId, resourceType: CloudinaryResourceType.image, invalidate: false);
  }
  Future<void> deleteHistory() async {

  }
}
