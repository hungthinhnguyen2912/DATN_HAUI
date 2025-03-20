import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:get/get.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:vegetable_classification/models/User.dart';
import 'package:vegetable_classification/views/auth/log_in_page.dart';
import 'package:vegetable_classification/views/home/home_page.dart';

class AuthController extends GetxController {
  final FirebaseAuth _auth = FirebaseAuth.instance;
  final GoogleSignIn _googleSignIn = GoogleSignIn();
  final FirebaseFirestore _firestore = FirebaseFirestore.instance;
  Rxn<User> firebaseUser = Rxn<User>();
  Rx<UserModel?> currentUser = Rx<UserModel?>(null);

  Future<void> loginWithGoogle() async {
    try {
      final GoogleSignInAccount? googleUser = await _googleSignIn.signIn();
      if (googleUser == null) return;

      final GoogleSignInAuthentication googleAuth =
          await googleUser.authentication;
      final AuthCredential credential = GoogleAuthProvider.credential(
        idToken: googleAuth.idToken,
        accessToken: googleAuth.accessToken,
      );
      UserCredential userCredential = await _auth.signInWithCredential(
        credential,
      );
      if (userCredential.additionalUserInfo!.isNewUser) {
        await _firestore.collection("User").doc(userCredential.user!.uid).set({
          "name": googleUser.displayName,
          "email": googleUser.email,
          "createdAt": DateTime.now(),
          "uid": userCredential.user!.uid,
        });
      } else {
        DocumentSnapshot userDoc =
            await _firestore
                .collection("User")
                .doc(userCredential.user!.uid)
                .get();
        currentUser.value = UserModel(
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl: userDoc.data().toString().contains('avatarUrl')
              ? userDoc['avatarUrl']
              : "",
        );
      }
    } on FirebaseAuthException catch (e) {
      if (e.code == 'account-exists-with-different-credential') {
        Get.snackbar("Error", "Tài khoản đã được đăng ký bằng phương thức khác.");
      } else {
        Get.snackbar("Error", "Có lỗi xảy ra: ${e.message}");
      }
    }
  }

  Future<void> login(String email, String password) async {
    try {
      await _auth.signInWithEmailAndPassword(email: email, password: password);
      DocumentSnapshot userDoc =
          await _firestore.collection("User").doc(_auth.currentUser!.uid).get();
      if (userDoc.exists && userDoc.data() != null) {
        currentUser.value = UserModel(
          uid: userDoc['uid'],
          name: userDoc['name'],
          email: userDoc['email'],
          createdAt: userDoc['createdAt'],
          avatarUrl: userDoc.data().toString().contains('avatarUrl')
              ? userDoc['avatarUrl']
              : "",
        );
        Get.off(HomePage());
      } else {
        Get.snackbar("Error", "Tài khoản không tồn tại.");
      }
    } on FirebaseAuthException {
      Get.snackbar("Error", "Something went wrong");
    }
  }

  Future<void> register(String email, String password, String name) async {
    try {
      UserCredential userCredential = await _auth
          .createUserWithEmailAndPassword(email: email, password: password);
      UserModel user = UserModel(
        name: name,
        email: email,
        createdAt: DateTime.now(),
        uid: userCredential.user!.uid,
      );
      await _firestore
          .collection("User")
          .doc(userCredential.user!.uid)
          .set(user.toJson());
      Get.off(LogInPage());
    } on FirebaseAuthException catch (e) {
      if (e.code == 'email-already-in-use') {
        Get.snackbar("Error", "Email này đã được sử dụng.");
      } else {
        Get.snackbar("Error", "Đăng ký thất bại: ${e.message}");
      }
    }
  }

  void logOut() async {
    await _auth.signOut();
    firebaseUser.value = null;
    currentUser.value = UserModel(
      uid: "",
      name: "",
      email: "",
      createdAt: DateTime.now(),
      avatarUrl: "",
    );
    Get.off(LogInPage());
  }
}
