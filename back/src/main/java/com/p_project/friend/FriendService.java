package com.p_project.friend;

import com.p_project.user.UserDTO;
import com.p_project.user.UserEntity;
import com.p_project.user.UserService;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Slf4j
@Service
@RequiredArgsConstructor
public class FriendService {

    private final FriendRepository friendRepository;
    private final UserService userService;

    public void addFriend(FriendDTO friendDTO){

        log.info("in FriendService: addFriend");
        Long friendId = Long.valueOf(userService.findByNickname(friendDTO.getFriendNickName())
                .orElseThrow(() -> new RuntimeException("User not found"))
                .getId());

        friendDTO.setToUserId(friendId);
        FriendEntity friendEntity = friendDTO.toEntity();
        friendRepository.save(friendEntity);
    }

    public List<UserDTO> getMutualFriends(Long userId) {
        List<UserEntity> users = friendRepository.findMutualFriends(userId);

        return users.stream()
                .map(UserDTO::fromEntity)
                .toList();
    }

    public List<UserDTO> getPendingRequests(Long userId) {
        List<UserEntity> users = friendRepository.findPendingRequestSenders(userId);

        return users.stream()
                .map(UserDTO::fromEntity)
                .toList();
    }

    @Transactional
    public void acceptFriend(Long fromUserId, Long toUserId) {
        if (friendRepository.existsFriendship(fromUserId, toUserId) < 1) {
            friendRepository.acceptFriendRequest(fromUserId, toUserId);
        }
    }

    @Transactional
    public void sendFriendRequest(Long fromUserId, String email) {
        friendRepository.sendFriendRequest(fromUserId, email);
    }

    @Transactional
    public void deleteFriendRequest(Long fromUserId, Long toUserId) {
        friendRepository.deleteFriendRequest(fromUserId, toUserId);
    }

}
