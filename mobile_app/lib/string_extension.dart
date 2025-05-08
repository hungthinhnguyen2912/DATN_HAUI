import 'dart:math';

String generateRandomString(int length) {
  const characters =
      'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  Random random = Random();
  StringBuffer buffer = StringBuffer();

  for (int i = 0; i < length; i++) {
    int randomIndex = random.nextInt(characters.length);
    buffer.write(characters[randomIndex]);
  }

  return buffer.toString();
}

String formatPhoneNumber(String phoneNumber) {
  if (phoneNumber.startsWith("0")) {
    return "+84${phoneNumber.substring(1)}";
  } else {
    return phoneNumber;
  }
}
